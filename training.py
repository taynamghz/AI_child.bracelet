from feature_extraction import extract_features
from model import train_model, save_model
from data_uploading import load_features

FEATURES_PATH = "/Users/taynamghz./Documents/bracelet/features.pkl"
folder_path = "/Users/taynamghz./Documents/bracelet"

def train_and_save():
    """Train the model and save it."""
    # Load features and labels from file
    features, labels = load_features(FEATURES_PATH)
    
    # If no features and labels found, perform feature extraction
    if not features and not labels:
        print("No features found. Extracting features...")
        features, labels = extract_features()  # This assumes extract_features() returns both features and labels.
    
    # Train the model
    model = train_model(features, labels)
    
    # Save the trained model
    save_model(model, folder_path)

# Run the training and saving process
train_and_save()