# Deep Learning Image Classification

This repository contains a TensorFlow implementation of Convolutional Neural Networks (CNNs) for image classification. It includes both a basic CNN architecture and a transfer learning approach using pretrained models.

## Features

- Basic CNN architecture with configurable layers
- Transfer learning support using Xception as base model
- Training visualization and model evaluation
- Support for both MNIST-like datasets and custom image datasets
- Data augmentation and preprocessing pipelines

## Requirements

- Python 3.7 or higher
- TensorFlow 2.8 or higher
- Additional dependencies listed in `requirements.txt`

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/deep-learning-classification.git
cd deep-learning-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic CNN Model

To train the basic CNN model on MNIST:

```bash
python train.py
```

This will:
- Load and preprocess the MNIST dataset
- Train the model for 10 epochs
- Save training history plots
- Save the trained model as 'mnist_classifier.h5'

### Transfer Learning

For transfer learning on custom datasets, modify the training script to use the TransferLearningClassifier:

```python
from model import TransferLearningClassifier

# Initialize model
model = TransferLearningClassifier(
    input_shape=(224, 224, 3),
    num_classes=your_num_classes
)

# Train initial layers
model.compile_model(learning_rate=0.1)
model.train(train_dataset, validation_data=valid_dataset, epochs=3)

# Fine-tune
model.unfreeze_top_layers(30)
model.compile_model(learning_rate=0.01)
model.train(train_dataset, validation_data=valid_dataset, epochs=10)
```

## Model Architecture

### Basic CNN
- 2 Convolutional layers with ReLU activation
- MaxPooling layer
- Batch Normalization
- Dropout layers for regularization
- Dense layers for classification

### Transfer Learning
- Pretrained Xception base
- Global Average Pooling
- Dense layer for classification

## Results

The basic CNN model achieves approximately 99% accuracy on the MNIST test set. Results may vary depending on the specific dataset and hyperparameters used.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on concepts from various deep learning resources and tutorials
- Uses TensorFlow and Keras frameworks
