# Gesture Recognition Assignment
This project implements a hybrid Convolutional Neural Network (CNN) and Recurrent Neural Network (RNN) model using Keras and TensorFlow for the classification of sequential image data. The model processes a sequence of images (like frames from a video) to predict one of several classes.

## Overview

The model architecture leverages both **CNN** for spatial feature extraction and **RNN** (via `ConvLSTM2D`) for capturing temporal dependencies between frames in a sequence. This architecture is beneficial for tasks that involve video or time-series image data.

### Model Architecture

- **TimeDistributed Conv2D**: The `Conv2D` layers are wrapped in a `TimeDistributed` layer to handle each image frame in a sequence independently.
- **ConvLSTM2D**: A ConvLSTM layer is added to capture the temporal dependencies between the image frames.
- **GlobalAveragePooling2D**: Used for dimensionality reduction by taking the average over the spatial dimensions.
- **Dense Layers**: Fully connected layers are used for final classification.

### Callbacks and Optimizer
- **ModelCheckpoint**: Saves the best model during training based on validation loss.
- **ReduceLROnPlateau**: Reduces the learning rate when the validation loss plateaus, to improve convergence.
- **Adam Optimizer**: Adam optimizer is used with an initial learning rate of `0.01`, which decreases over training.

## Requirements

The following packages are required:

- `tensorflow` (for Keras and training the model)
- `numpy` (for data manipulation)
- `opencv-python` (for handling image sequences)
- `matplotlib` (for visualizing training history)
- `scikit-learn` (for splitting data and metrics)

## Contact
Created by [@jnangineni] - feel free to contact me!
