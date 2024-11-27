import tensorflow as tf
import joblib
import pickle

# Load pre-trained model
with open('trained_model.pkl', 'rb') as model_file:
    trained_model = pickle.load(model_file)

# Load feature scaler
scaler = joblib.load('scaler.pkl')

# Dummy data to define input shape (replace with actual feature shape)
input_shape = (1, scaler.n_features_in_)

# Build a TensorFlow Keras model with the same architecture as the trained model
tf_model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(input_shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(trained_model.classes_), activation='softmax')
])

# Transfer weights from pre-trained model
tf_model.layers[1].set_weights([trained_model.coefs_[0], trained_model.intercepts_[0]])
tf_model.layers[2].set_weights([trained_model.coefs_[1], trained_model.intercepts_[1]])
tf_model.layers[3].set_weights([trained_model.coefs_[2], trained_model.intercepts_[2]])

# Save TensorFlow model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
tflite_model = converter.convert()

# Write the model to a .tflite file
with open('trained_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("TensorFlow Lite model saved as 'trained_model.tflite'")
