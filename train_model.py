import os
import numpy as np
import pandas as pd
import librosa
import joblib
import kagglehub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

def download_dataset():
    """Download RAVDESS dataset using kagglehub."""
    print("Downloading RAVDESS dataset...")
    try:
        # Download latest version
        path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
        print(f"Dataset downloaded successfully to: {path}")
        return path
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        return None

def extract_features(file_path):
    """Extract MFCC features from audio file."""
    try:
        # Load audio file
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        
        # Take mean of MFCCs
        mfccs_mean = np.mean(mfccs.T, axis=0)
        
        return mfccs_mean
    except Exception as e:
        print(f"Error extracting features from {file_path}: {str(e)}")
        return None

def prepare_dataset(data_path):
    """Prepare dataset from RAVDESS audio files."""
    features = []
    labels = []
    
    # RAVDESS emotion mapping
    emotion_map = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }
    
    # Walk through the data directory
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                
                # Extract emotion from filename (RAVDESS format)
                emotion = emotion_map[file.split('-')[2]]
                
                # Extract features
                feature = extract_features(file_path)
                if feature is not None:
                    features.append(feature)
                    labels.append(emotion)
    
    return np.array(features), np.array(labels)

def train_model():
    """Train and save the emotion detection model."""
    # Download dataset if not already present
    data_path = download_dataset()
    if data_path is None:
        print("Failed to download dataset. Please check your internet connection and try again.")
        return
    
    print("Loading and processing dataset...")
    X, y = prepare_dataset(data_path)
    
    if len(X) == 0 or len(y) == 0:
        print("No valid audio files found in the dataset.")
        return
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest classifier
    print("Training Random Forest classifier...")
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = rf_classifier.predict(X_test_scaled)
    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred))
    
    # Save the model and scaler
    print("\nSaving model and scaler...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(rf_classifier, 'models/emotion_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Model and scaler saved successfully!")

if __name__ == "__main__":
    train_model() 