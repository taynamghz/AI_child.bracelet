import librosa
import numpy as np
import os

def fix_length(audio_data, sampling_rate, duration=2):
    """Fix audio length to exactly `duration` seconds."""
    desired_samples = sampling_rate * duration
    if len(audio_data) < desired_samples:
        audio_data = np.pad(audio_data, (0, desired_samples - len(audio_data)), 'constant')
    elif len(audio_data) > desired_samples:
        audio_data = audio_data[:desired_samples]
    return audio_data

def extract_audio_features(file_path, sampling_rate=16000):
    """Extract audio features from a given file path, fixed to a certain length."""
    try:
        # Load the audio file using librosa
        y, sr = librosa.load(file_path, sr=sampling_rate)  # y = audio time series, sr = sample rate
        
        # Fix audio length to the desired duration
        y = fix_length(y, sampling_rate)
        
        # Extract MFCC features (13 MFCC coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=27)
        print(mfcc)
        # Return the mean of the MFCCs across time as the feature vector
        return np.mean(mfcc, axis=1)
    
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

def extract_features_from_directory(directory_path, label, sampling_rate=16000):
    """Extract features from all audio files in a given directory."""
    features = []
    labels = []
    
    # Iterate through all files in the directory
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".wav"):
            file_path = os.path.join(directory_path, file_name)
            feature_vector = extract_audio_features(file_path, sampling_rate)
            
            if feature_vector is not None:
                features.append(feature_vector)
                labels.append(label)  # Assign the label (positive/negative)
    
    # Convert lists to numpy arrays and return
    return np.array(features), np.array(labels)
def extract_features(audio_data, sampling_rate=16000):
    """Extract features (e.g., MFCCs) from real-time audio data."""
    try:
        # Fix the length of audio to the desired duration
        audio_data = fix_length(audio_data, sampling_rate)
        
        # Extract 27 MFCC features (no delta or delta-delta features)
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sampling_rate, n_mfcc=27)
        
        # Return the mean of the MFCCs across time as the feature vector
        return np.mean(mfcc, axis=1)  # This will return a vector of length 27
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None
# Example usage:
positive_features, positive_labels = extract_features_from_directory(
    '/Users/taynamghz./Documents/bracelet/trainingdata/positive', 1)  # 1 for positive class

negative_features, negative_labels = extract_features_from_directory(
    '/Users/taynamghz./Documents/bracelet/trainingdata/negative', 0)  # 0 for negative class

# Combine the positive and negative features
X = np.vstack([positive_features, negative_features])
y = np.concatenate([positive_labels, negative_labels])

# Print the shapes of the feature matrix and labels
print(f"Feature matrix X shape: {X.shape}")
print(f"Labels vector y shape: {y.shape}")
