
# README for Exercise Classification Model Project

## Overview
This project focuses on developing a machine learning model that uses both spatial (image data) and temporal (numerical metadata) inputs to classify exercises into categories such as push-ups, pull-ups, and sit-ups. The model architecture includes convolutional layers, LSTM units, and dense layers to effectively process and integrate different types of data for accurate classification.

## Model and Code Overview

### `Model.py`
This script contains the entire model workflow including data loading, preprocessing, model building, training, and evaluation. Here's a breakdown of its functionality:

#### Data Loading and Preprocessing
- **Images and Metadata Loading**: Functions to load and preprocess image sequences and metadata from specified directories.
- **Sequence Generation**: Converts lists of image paths and metadata into sequential data suitable for LSTM processing.

#### Model Architecture
- **Input Layers**: Separate input layers for spatial and temporal data.
- **Convolutional Layers**: Applied to spatial inputs to extract feature maps.
- **LSTM Layers**: Used to process both spatial and temporal sequences.
- **Concatenation and Output**: Combines features from both branches and passes them through dense layers for final classification.

#### Model Training and Evaluation
- **Compilation and Fitting**: Model is compiled with Adam optimizer and trained using early stopping.
- **Performance Metrics**: Outputs classification report and confusion matrix to evaluate model accuracy.

### Usage
To run the model training and evaluation, execute the script as follows:
\```bash
python Model.py
\```

### Note
Ensure you have TensorFlow, NumPy, OpenCV, Pandas, Matplotlib, and Seaborn installed. Adjust paths to your data directories as necessary.

## Setup and Installation

1. Ensure Python 3.8+ is installed on your system.
2. Install the required libraries using:
   \```bash
   pip install numpy pandas tensorflow opencv-python-headless matplotlib seaborn
   \```

3. Download the dataset and organize it into the required directory structure as mentioned in the script comments.
4. Run the script from your terminal or an IDE like PyCharm or VS Code.

## Model Details and Parameters

### Model Parameters
- **Conv2D Layers**: Filters=32, Kernel Size=3x3
- **LSTM Units**: Spatial=64, Temporal=32
- **Dense Layers**: Units=64 with ReLU activation, followed by a dropout of 0.5
- **Output Layer**: Softmax activation for classification

### Training Configuration
- **Optimizer**: Adam with a learning rate of 0.0001
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy

## Contributions
Contributions to this project are welcome! Please fork this repository, make your changes, and submit a pull request.

## License
This project is released under the MIT License.
