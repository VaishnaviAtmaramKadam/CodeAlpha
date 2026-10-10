# Handwritten Character Recognition

## Project Overview

This project recognizes handwritten digits from 0 to 9 using a Convolutional Neural Network (CNN).

## Objective

To identify handwritten digits using image processing and deep learning.

## Dataset

* MNIST handwritten digit dataset
* 60,000 training images
* 10,000 testing images
* Image size: 28 × 28 pixels

## Technologies Used

* Python
* TensorFlow and Keras
* NumPy
* Matplotlib
* Scikit-learn

## Model Architecture

* Convolutional layers
* Max pooling layers
* Flatten layer
* Dense layer
* Dropout layer
* Softmax output layer

## Results

* Test Accuracy: 98.93%
* Training Epochs: 5

## Output Files

* `handwritten_character_model.keras`
* `training_accuracy.png`
* `sample_predictions.png`

## How to Run

1. Install the required libraries:
   `pip install tensorflow matplotlib scikit-learn`
2. Run the program:
   `python handwritten_character_recognition.py`

## Conclusion

The CNN model successfully recognizes handwritten digits from the MNIST dataset with 98.93% test accuracy.
