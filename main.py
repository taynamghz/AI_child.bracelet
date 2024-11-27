import joblib
import numpy as np
import sounddevice as sd
from feature_extraction import extract_features  # Assuming this function is correctly defined

# Function to load the trained model
def load_model(model_path):
    model = joblib.load(model_path)  # Use joblib to load the model
    return model

def main():
    model_path = "/Users/taynamghz./Documents/bracelet/trained_model.pkl"  # Correct path to the model file
    print(f"Loading model from {model_path}...")
    speaker_model = load_model(model_path)
        
    threshold = 0.5
    

    
     # Lower the threshold
    samplerate = 16000  # Sample rate for microphone
    
    # Buffer to collect enough audio data before processing
    duration = 5
     # Duration in seconds for feature extraction

    audio_buffer = np.zeros(samplerate * duration)

    # Callback function to process the audio from the microphone
    def process_audio(indata, frames, time, status):
        if status:
            print(status)
        
        # Append new audio data to the buffer
        nonlocal audio_buffer
        audio_buffer = np.roll(audio_buffer, -frames)
        audio_buffer[-frames:] = indata[:, 0]

        # Check if the buffer has enough data for processing
        if len(audio_buffer) >= samplerate * duration:
            # Extract features from the buffered audio
            features = extract_features(audio_buffer, samplerate)
            
            # Print the shape of the features
            print(f"Extracted features shape: {features.shape}")  # Should be (27,) if expected
            
            if features is not None:
                # Reshape the features to match the model's expected input shape (1, 27)
                features_reshaped = features.reshape(1, -1)  # This reshapes it to (1, 27)
                
                # Make a prediction using the trained model
                prediction_proba = speaker_model.predict_proba(features_reshaped)
                
                print(f"Prediction probabilities: {prediction_proba[0]}")  # Show probabilities

                # Check if the model predicts the voice as "Taynam"
                if prediction_proba[0][1] > threshold:
                    print("Taynam: Voice recognized!")
                else:
                    print("Not Taynam: Voice not recognized.")

    # Initialize the microphone input stream
    with sd.InputStream(channels=1, samplerate=samplerate, callback=process_audio):
        print("Listening...")
        input("Press Enter to stop.")  # Keep the program running until Enter is pressed

if __name__ == "__main__":
    main()

