import os
import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Dropout, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Helper function to load and preprocess images
def load_and_preprocess_image(image_path, img_size=(128, 128)):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Failed to load image from {image_path}")
        return None
    image = cv2.resize(image, img_size)
    return image.astype('float32') / 255.0

# Generate sequences from lists of image paths
def create_sequences(image_paths, sequence_length):
    sequences = []
    for i in range(0, len(image_paths) - sequence_length + 1, sequence_length):
        sequence = [load_and_preprocess_image(img_path) for img_path in image_paths[i:i + sequence_length] if img_path is not None]
        if len(sequence) == sequence_length:
            sequences.append(np.array(sequence))
    return sequences

# Create temporal data sequences from metadata
def create_temporal_sequences(metadata_dir, sequence_length):
    if not os.path.exists(metadata_dir):
        print(f"Directory does not exist: {metadata_dir}")
        return []
    metadata_paths = sorted([os.path.join(metadata_dir, f) for f in os.listdir(metadata_dir) if f.endswith('.csv')])
    sequences = []
    for i in range(0, len(metadata_paths) - sequence_length + 1, sequence_length):
        sequence_data = []
        for file_path in metadata_paths[i:i + sequence_length]:
            df = pd.read_csv(file_path).select_dtypes(include=[np.number]).astype(np.float32)
            sequence_data.append(df.values.flatten())
        sequences.append(np.stack(sequence_data))
    return sequences

# Define the model architecture
def build_model(spatial_shape, temporal_shape, num_classes):
    spatial_input = Input(shape=spatial_shape)
    x = TimeDistributed(Conv2D(32, (3, 3), activation='relu'))(spatial_input)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = TimeDistributed(Flatten())(x)
    spatial_output = LSTM(64)(x)

    temporal_input = Input(shape=(None, temporal_shape[-1]))
    y = LSTM(32)(temporal_input)
    temporal_output = Dense(64, activation='relu')(y)

    combined = concatenate([spatial_output, temporal_output])
    z = Dense(64, activation='relu')(combined)
    z = Dropout(0.5)(z)
    outputs = Dense(num_classes, activation='softmax')(z)

    model = Model(inputs=[spatial_input, temporal_input], outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    base_dir = '/home/user6/HSiPu2/HSiPu2'
    sub_dirs = ['fwcce', 'fwcz', 'ytc', 'ytz', 'ywqc', 'ywqzz']
    sequence_length = 10

    all_spatial_sequences = []
    all_temporal_sequences = []
    labels = []

    for sub_dir in sub_dirs:
        print(f"Processing directory: {sub_dir}")
        image_dir = os.path.join(base_dir, 'all_pictures', sub_dir)
        image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')])
        spatial_sequences = create_sequences(image_files, sequence_length)
        all_spatial_sequences.extend(spatial_sequences)

        standard_dir = os.path.join(base_dir, 'standard', sub_dir)
        not_standard_dir = os.path.join(base_dir, 'not_standard', sub_dir)
        standard_temporal_sequences = create_temporal_sequences(standard_dir, sequence_length)
        not_standard_temporal_sequences = create_temporal_sequences(not_standard_dir, sequence_length)
        all_temporal_sequences.extend(standard_temporal_sequences)
        all_temporal_sequences.extend(not_standard_temporal_sequences)

        labels.extend([sub_dirs.index(sub_dir) // 2] * len(spatial_sequences))

    labels = to_categorical(labels, num_classes=3)

    min_length = min(len(all_spatial_sequences), len(all_temporal_sequences))
    all_spatial_sequences = all_spatial_sequences[:min_length]
    all_temporal_sequences = all_temporal_sequences[:min_length]
    labels = labels[:min_length]

    X_spatial_temp, X_spatial_test, X_temporal_temp, X_temporal_test, y_temp, y_test = train_test_split(
        all_spatial_sequences, all_temporal_sequences, labels, test_size=0.2, random_state=42)
    X_spatial_train, X_spatial_val, X_temporal_train, X_temporal_val, y_train, y_val = train_test_split(
        X_spatial_temp, X_temporal_temp, y_temp, test_size=0.25, random_state=42)

    model = build_model(X_spatial_train[0].shape, X_temporal_train[0].shape, 3)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        [X_spatial_train, X_temporal_train],
        y_train,
        epochs=40,
        batch_size=2,
        validation_data=([X_spatial_val, X_temporal_val], y_val),
        callbacks=[early_stopping]
    )

    model_path = os.path.join(base_dir, 'saved_model.h5')
    model.save(model_path)
    print(f"Model saved to {model_path}")

    final_val_loss = history.history['val_loss'][-1]
    final_val_accuracy = history.history['val_accuracy'][-1]
    print(f"Final Validation Loss: {final_val_loss:.4f}, Final Validation Accuracy: {final_val_accuracy:.4f}")

    test_results = model.evaluate([X_spatial_test, X_temporal_test], y_test)
    print("\nTest Loss:", test_results[0])
    print("Test Accuracy:", test_results[1])

    # Check the shape of temporal sequences before and after padding
    X_temporal_test_np = np.array(X_temporal_test)
    print("Shape of X_temporal_test before padding:", X_temporal_test_np.shape)
    # Pad temporal sequences to have a fixed length
    max_temporal_length = max(len(seq) for seq in X_temporal_test_np)
    X_temporal_test_padded = tf.keras.preprocessing.sequence.pad_sequences(X_temporal_test_np, maxlen=max_temporal_length, padding='post')
    print("Shape of X_temporal_test after padding:", X_temporal_test_padded.shape)
    # Predict on the test set
    batch_size = len(X_spatial_test)
    y_pred = model.predict([X_spatial_test, X_temporal_test_padded], batch_size=batch_size)

    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    report_str = classification_report(y_true_classes, y_pred_classes, target_names=["Class 0", "Class 1", "Class 2"])
    print("Classification Report:\n", report_str)

    # Confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Class 0", "Class 1", "Class 2"], yticklabels=["Class 0", "Class 1", "Class 2"])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # Optionally save the plot
    plt.savefig(os.path.join(base_dir, 'confusion_matrix.png'))

    # Saving the classification report to a text file
    report_path = os.path.join('/home/user6/HSiPu2/HSiPu2/', 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_str)
    print(f"Classification report saved to {report_path}")