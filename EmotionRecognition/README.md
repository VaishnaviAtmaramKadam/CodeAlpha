# Emotion Recognition from Speech

## Project Overview

This project recognizes human emotions from speech audio using Machine Learning.

## Dataset

RAVDESS Speech Emotion Dataset containing 1,440 audio files.

## Features

* Audio feature extraction using MFCC
* Random Forest Classification
* Emotion classification report
* Emotion distribution graph

## Emotions

* Neutral
* Calm
* Happy
* Sad
* Angry
* Fearful
* Disgust
* Surprised

## Model Performance

Model accuracy: 61.81%

## Technologies Used

* Python
* Librosa
* NumPy
* Scikit-learn
* Matplotlib
* Joblib

## How to Run

1. Install the required Python libraries.
2. Place the dataset inside the `dataset` folder.
3. Run `python emotion_recognition.py`.

## Output

The trained model is saved as `emotion_model.pkl`, and the emotion distribution graph is saved as `emotion_distribution.png`.
