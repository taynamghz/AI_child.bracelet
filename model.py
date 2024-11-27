import os
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier

FEATURES_PATH = "/Users/taynamghz./Documents/bracelet/features.pkl"
folder_path = "/Users/taynamghz./Documents/bracelet"

def load_features(features_path):
    """Load the features from the pkl file."""
    if os.path.exists(features_path):
        with open(features_path, 'rb') as f:
            features, labels = joblib.load(f)
        print(f"Features and labels loaded from {features_path}")
    else:
        print(f"Features file not found at {features_path}. Initializing with empty data.")
        features, labels = [], []  # Initialize with empty data if the file doesn't exist
    return features, labels

def save_features(features, labels, features_path):
    """Save the features and labels to the pkl file."""
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    with open(features_path, 'wb') as f:
        joblib.dump((features, labels), f)
    print(f"Features and labels saved to {features_path}")

def train_model(features, labels):
    """Train an MLPClassifier."""
    if len(features) == 0 or len(labels) == 0:
        print("Error: No features or labels to train on.")
        return None
    print("Training model...")
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    model.fit(features, labels)
    print("Model training complete.")
    return model

def save_model(model, folder_path):
    """Save the trained model."""
    if model is None:
        print("No model to save. Training failed.")
        return
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, "trained_model.pkl")
    joblib.dump(model, file_path)
    print(f"Model saved at {file_path}")

def load_model(folder_path):
    """Load a trained model."""
    file_path = os.path.join(folder_path, "trained_model.pkl")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No model found at {file_path}")
    return joblib.load(file_path)
