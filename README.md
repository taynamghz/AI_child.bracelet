# AI Child Bracelet - Assistive Wearable for Children with ASD & ADHD


# Overview

The AI Child Bracelet is a wearable device designed to assist children with Autism Spectrum Disorder (ASD) and Attention Deficit Hyperactivity Disorder (ADHD) during hyperfocus episodes. By using a machine learning-based voice recognition system, it detects specific voices (e.g., caregivers or teachers) and provides gentle vibration cues to help redirect the child’s attention, improving situational awareness without forceful disruptions.

# Features

Voice Recognition: Detects specific voices and activates responses.
Gentle Vibration Cues: Non-invasive cues to help redirect attention.
Machine Learning Model: Trained to identify and react to caregiver and teacher voices.
Wearable Design: Comfortable and portable for everyday use.
Improved Engagement: Helps children re-engage with their environment by reducing isolation during hyperfocus.
Files and Code Explanation

# 1. main.py
This is the main file that runs the wearable device's logic. It integrates the machine learning model for voice recognition, processes input data, and triggers the vibration motor to provide subtle feedback to the child. It serves as the heart of the device’s functionality.

# 2. voice_recognition.py
This file handles the voice recognition functionality. Using a trained machine learning model, it listens for specific voices (e.g., caregivers or teachers). Once a voice is detected, the system activates a predefined response (vibration cue). This script may rely on libraries like speech_recognition or a custom trained model using TensorFlow.

# 3. vibration_feedback.py
Responsible for controlling the hardware for vibration feedback. This file communicates with the bracelet’s motor, activating it based on the voice detection events. It ensures the vibration is subtle and non-intrusive, just enough to reorient the child’s attention without overwhelming them.

# 4. model_training.py
This script is used to train the machine learning model for voice recognition. It collects voice data, processes it, and trains a model to distinguish between different voices. The trained model is then saved for real-time use by the wearable device.

# 5. README.md
This file! It provides an overview of the project, explains the codebase, installation instructions, and usage. This is where users will get all the essential information about the project.

# Installation Instructions
To get started with the AI Child Bracelet, clone the repository and install the necessary dependencies:

Clone the repository:

git clone https://github.com/yourusername/AI_child.bracelet.git


Install required libraries:


pip install -r requirements.txt


Run the main script:


python main.py

