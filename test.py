import joblib
import librosa
import numpy as np
from feature_extraction import extract_features

# Function to predict whether the audio is positive or negative
def predict_audio(file_path, model_path="trained_model.pkl"):
    try:
        # Load the trained model
        model = joblib.load(model_path)
        
        # Load the audio file
        audio, sr = librosa.load(file_path, sr=None)
        if audio is None:
            return "Error: Could not load audio file"
        
        # Convert audio to floating-point format if needed
        if not np.issubdtype(audio.dtype, np.floating):
            audio = audio.astype(np.float32)
        
        # Extract features from the audio file
        features = extract_features(audio, sr)
        if features is None:
            return "Error: Could not extract features"
        
        # Reshape features for prediction
        features = features.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)
        return "Positive" if prediction[0] == 1 else "Negative"
    except Exception as e:
        return f"Error during prediction: {e}"

# Main function to test two audio files
if __name__ == "__main__":
    # Paths to test audio files
    audio_file_1 = "test/1.wav"
    audio_file_2 = "test/2.wav"
    
    # Path to the trained model
    model_path = "trained_model.pkl"
    
    # Test the first audio file
    print(f"Testing {audio_file_1}:")
    result_1 = predict_audio(audio_file_1, model_path)
    print(f"Result: {result_1}")
    
    # Test the second audio file
    print(f"\nTesting {audio_file_2}:")
    result_2 = predict_audio(audio_file_2, model_path)
    print(f"Result: {result_2}")
