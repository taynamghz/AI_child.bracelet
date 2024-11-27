import os
import librosa
import numpy as np
import pickle
from tqdm import tqdm

def load_audio_file(file_path):
    """Load audio data from a file."""
    try:
        # Load audio with librosa, enforcing floating-point format
        audio, sr = librosa.load(file_path, sr=None)
        return audio, sr
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None, None

def fix_length(audio_data, sampling_rate, duration=2):
    """Fix audio length to exactly `duration` seconds."""
    desired_samples = sampling_rate * duration
    if len(audio_data) < desired_samples:
        audio_data = np.pad(audio_data, (0, desired_samples - len(audio_data)), 'constant')
    elif len(audio_data) > desired_samples:
        audio_data = audio_data[:desired_samples]
    return audio_data

def extract_features(audio, sr=16000, n_mfcc=13):
    """Extract MFCC, delta MFCC, and energy features."""
    try:
        audio = fix_length(audio, sr, 2)  # Fix length to 2 seconds
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        delta_mfcc = librosa.feature.delta(mfccs)
        energy = librosa.feature.rms(y=audio)
        # Ensure the feature vectors are consistently sized
        features = np.hstack([mfccs.mean(axis=1), delta_mfcc.mean(axis=1), energy.mean()])
        return features
    except Exception as e:
        print(f"Error in feature extraction: {e}")
        return None

def load_audio_files_from_directory(directory, label):
    """Load audio files from a directory and assign them a label."""
    audio_data, labels = [], []
    for file in os.listdir(directory):
        if file.endswith(".wav"):
            audio, _ = load_audio_file(os.path.join(directory, file))
            if audio is not None:
                audio_data.append(audio)
                labels.append(label)
    return audio_data, labels

def load_training_data_separately(data_directory):
    """Load positive and negative audio data and labels separately."""
    positive_dir = os.path.join(data_directory, 'positive')
    negative_dir = os.path.join(data_directory, 'negative')

    print("Loading positive samples...")
    positive_audio, positive_labels = load_audio_files_from_directory(positive_dir, label=1)

    print("Loading negative samples...")
    negative_audio, negative_labels = load_audio_files_from_directory(negative_dir, label=0)

    # Combine positive and negative samples
    audio_data = positive_audio + negative_audio
    labels = positive_labels + negative_labels

    return audio_data, labels

def process_audio_files(data_directory, output_file='features.pkl'):
    """Process positive and negative audio files separately and save to a pickle file."""
    audio_files, labels = load_training_data_separately(data_directory)

    features = []
    valid_labels = []
    for audio, label in tqdm(zip(audio_files, labels), desc="Extracting features"):
        feature = extract_features(audio)
        if feature is not None:
            features.append(feature)
            valid_labels.append(label)

    # Save features and labels
    with open(output_file, 'wb') as f:
        pickle.dump((features, valid_labels), f)

    print(f"Features and labels saved to {output_file}")
    
def load_features(file_path):
    """Load features and labels from a pickle file."""
    try:
        with open(file_path, 'rb') as f:
            features, labels = pickle.load(f)
        print(f"Loaded features from {file_path}.")
        return features, labels
    except FileNotFoundError:
        print(f"File not found: {file_path}. Returning empty lists.")
        return [], []
    except Exception as e:
        print(f"Error loading features from {file_path}: {e}")
        return [], []
    
if __name__ == "__main__":
    # Update this to the actual path of your training data directory
    data_directory = "/Users/taynamghz./Documents/bracelet/trainingdata"
    output_file = "/Users/taynamghz./Documents/bracelet/features.pkl"

    process_audio_files(data_directory, output_file)
