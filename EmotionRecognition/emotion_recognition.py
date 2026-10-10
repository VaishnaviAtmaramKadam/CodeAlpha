
import os
from pathlib import Path

import numpy as np
import librosa
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Dataset folder
DATASET_PATH = Path("dataset")

# RAVDESS emotion labels
EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

def extract_features(file_path):
    audio, sample_rate = librosa.load(
        file_path, sr=22050, duration=3
    )

    mfcc = librosa.feature.mfcc(
        y=audio, sr=sample_rate, n_mfcc=40
    )

    return np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1)
    ])

features = []
labels = []

print("Reading audio files...")

for file_path in DATASET_PATH.rglob("*.wav"):
    parts = file_path.stem.split("-")

    if len(parts) != 7:
        continue

    emotion_code = parts[2]

    if emotion_code not in EMOTIONS:
        continue

    try:
        feature = extract_features(str(file_path))
        features.append(feature)
        labels.append(EMOTIONS[emotion_code])
    except Exception as error:
        print(f"Skipped {file_path.name}: {error}")

if not features:
    raise RuntimeError(
        "No audio files found. Check the dataset folder location."
    )

X = np.array(features)
y = np.array(labels)

print("Total audio files:", len(X))

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

print("Training the model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0
    )
)

# Save the trained model
import joblib

joblib.dump(
    {"model": model, "encoder": encoder},
    "emotion_model.pkl"
)

print("\nModel saved as emotion_model.pkl")


# Plot emotion distribution
unique_emotions, counts = np.unique(y, return_counts=True)

plt.figure(figsize=(9, 5))
plt.bar(unique_emotions, counts)
plt.title("Emotion Distribution in the Dataset")
plt.xlabel("Emotion")
plt.ylabel("Number of Audio Files")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("emotion_distribution.png")
plt.close()

print("\nGraph saved as emotion_distribution.png")
print("Task 2 completed successfully!")