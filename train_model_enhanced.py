import os
import numpy as np
import pandas as pd
import librosa
import joblib
import kagglehub
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings("ignore")

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

def extract_comprehensive_features(file_path):
    """Extract comprehensive audio features for better emotion recognition."""
    try:
        # Load audio file with better preprocessing
        audio, sample_rate = librosa.load(file_path, sr=22050, res_type='kaiser_fast')
        
        # Preprocessing: trim silence and normalize
        audio, _ = librosa.effects.trim(audio, top_db=20)
        audio = librosa.util.normalize(audio)
        
        features = []
        
        # 1. MFCC Features (enhanced)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs_std = np.std(mfccs.T, axis=0)
        features.extend(mfccs_mean)
        features.extend(mfccs_std)
        
        # 2. Delta and Delta-Delta MFCCs
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        features.extend(np.mean(delta_mfccs.T, axis=0))
        features.extend(np.mean(delta2_mfccs.T, axis=0))
        
        # 3. Spectral Features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
        features.append(np.mean(spectral_centroids))
        features.append(np.std(spectral_centroids))
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0]
        features.append(np.mean(spectral_rolloff))
        features.append(np.std(spectral_rolloff))
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)[0]
        features.append(np.mean(spectral_bandwidth))
        features.append(np.std(spectral_bandwidth))
        
        # 4. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features.append(np.mean(zcr))
        features.append(np.std(zcr))
        
        # 5. Chroma Features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        features.extend(np.mean(chroma.T, axis=0))
        features.extend(np.std(chroma.T, axis=0))
        
        # 6. Tonnetz Features
        tonnetz = librosa.feature.tonnetz(y=audio, sr=sample_rate)
        features.extend(np.mean(tonnetz.T, axis=0))
        
        # 7. Spectral Contrast
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        features.extend(np.mean(contrast.T, axis=0))
        features.extend(np.std(contrast.T, axis=0))
        
        # 8. Mel Spectrogram Features
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
        features.append(np.mean(mel_spec))
        features.append(np.std(mel_spec))
        
        # 9. RMS Energy
        rms = librosa.feature.rms(y=audio)[0]
        features.append(np.mean(rms))
        features.append(np.std(rms))
        
        # 10. Tempo and Rhythm Features
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sample_rate)
        features.append(tempo)
        
        # 11. Harmonic and Percussive Components
        harmonic, percussive = librosa.effects.hpss(audio)
        features.append(np.mean(harmonic))
        features.append(np.mean(percussive))
        
        # 12. Spectral Flatness
        flatness = librosa.feature.spectral_flatness(y=audio)[0]
        features.append(np.mean(flatness))
        features.append(np.std(flatness))
        
        # 13. Poly Features
        poly_features = librosa.feature.poly_features(y=audio, sr=sample_rate)
        features.extend(np.mean(poly_features.T, axis=0))
        
        return np.array(features)
        
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
    
    print("Processing audio files...")
    processed_count = 0
    
    # Walk through the data directory
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                
                try:
                    # Extract emotion from filename (RAVDESS format)
                    emotion = emotion_map[file.split('-')[2]]
                    
                    # Extract comprehensive features
                    feature = extract_comprehensive_features(file_path)
                    if feature is not None and len(feature) > 0:
                        features.append(feature)
                        labels.append(emotion)
                        processed_count += 1
                        
                        if processed_count % 100 == 0:
                            print(f"Processed {processed_count} files...")
                            
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")
                    continue
    
    print(f"Successfully processed {processed_count} audio files")
    return np.array(features), np.array(labels)

def train_enhanced_model():
    """Train and save an enhanced emotion detection model."""
    # Download dataset if not already present
    data_path = download_dataset()
    if data_path is None:
        print("Failed to download dataset. Please check your internet connection and try again.")
        return
    
    print("Loading and processing dataset with enhanced features...")
    X, y = prepare_dataset(data_path)
    
    if len(X) == 0 or len(y) == 0:
        print("No valid audio files found in the dataset.")
        return
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Classes: {np.unique(y)}")
    
    # Check for any NaN or infinite values
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        print("Warning: Found NaN or infinite values in features. Replacing with 0.")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Enhanced model pipeline with feature selection
    pipe = Pipeline([
        ('scaler', RobustScaler()),  # More robust to outliers than StandardScaler
        ('feature_selection', SelectKBest(f_classif, k=min(200, X.shape[1]))),  # Select best features
        ('clf', RandomForestClassifier(random_state=42))
    ])
    
    # Comprehensive parameter grid for better models
    param_grid = [
        {
            'clf': [RandomForestClassifier(random_state=42)],
            'clf__n_estimators': [300, 500, 800],
            'clf__max_depth': [None, 30, 50],
            'clf__min_samples_split': [2, 5],
            'clf__min_samples_leaf': [1, 2],
            'clf__max_features': ['sqrt', 'log2']
        },
        {
            'clf': [GradientBoostingClassifier(random_state=42)],
            'clf__n_estimators': [200, 300],
            'clf__learning_rate': [0.05, 0.1, 0.2],
            'clf__max_depth': [3, 5, 7],
            'clf__subsample': [0.8, 1.0]
        },
        {
            'clf': [ExtraTreesClassifier(random_state=42)],
            'clf__n_estimators': [300, 500],
            'clf__max_depth': [None, 30, 50],
            'clf__min_samples_split': [2, 5],
            'clf__min_samples_leaf': [1, 2]
        },
        {
            'clf': [SVC(probability=True, random_state=42)],
            'clf__C': [1, 10, 100],
            'clf__kernel': ['rbf', 'poly'],
            'clf__gamma': ['scale', 'auto'],
            'clf__degree': [2, 3]
        },
        {
            'clf': [MLPClassifier(random_state=42, max_iter=1000)],
            'clf__hidden_layer_sizes': [(100, 50), (200, 100), (300, 150)],
            'clf__activation': ['relu', 'tanh'],
            'clf__alpha': [0.0001, 0.001, 0.01],
            'clf__learning_rate': ['constant', 'adaptive']
        }
    ]
    
    # Use fewer CV folds for faster training but still robust
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    print("Searching for the best model with enhanced features...")
    search = GridSearchCV(
        pipe, param_grid, cv=cv, n_jobs=-1, 
        scoring='accuracy', refit=True, verbose=1
    )
    
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    
    print(f'\nBest model: {best_model.named_steps["clf"].__class__.__name__}')
    print(f'Best parameters: {search.best_params_}')
    print(f'Best CV score: {search.best_score_:.4f}')

    # Evaluate on hold-out test set
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)
    
    print("\n" + "="*50)
    print("ENHANCED MODEL EVALUATION")
    print("="*50)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Test Accuracy: {accuracy:.4f}')
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Calculate confidence statistics
    max_probs = np.max(y_pred_proba, axis=1)
    print(f"\nConfidence Statistics:")
    print(f"Mean confidence: {np.mean(max_probs):.4f}")
    print(f"Median confidence: {np.median(max_probs):.4f}")
    print(f"Min confidence: {np.min(max_probs):.4f}")
    print(f"Max confidence: {np.max(max_probs):.4f}")
    print(f"High confidence predictions (>0.8): {np.sum(max_probs > 0.8)}/{len(max_probs)} ({np.sum(max_probs > 0.8)/len(max_probs)*100:.1f}%)")
    
    # Confusion Matrix
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Save the enhanced model
    print("\nSaving enhanced model and scaler...")
    os.makedirs('models', exist_ok=True)
    
    # Save components separately
    fitted_scaler = best_model.named_steps['scaler']
    fitted_feature_selector = best_model.named_steps['feature_selection']
    fitted_clf = best_model.named_steps['clf']
    
    joblib.dump(fitted_clf, 'models/emotion_model.pkl')
    joblib.dump(fitted_scaler, 'models/scaler.pkl')
    joblib.dump(fitted_feature_selector, 'models/feature_selector.pkl')
    
    print("Enhanced model, scaler, and feature selector saved successfully!")
    print(f"Model will now use {fitted_feature_selector.k_} best features out of {X.shape[1]} total features")
    
    return best_model, accuracy

if __name__ == "__main__":
    print("Training Enhanced SentiSound Model")
    print("="*50)
    model, accuracy = train_enhanced_model()
    print(f"\nTraining completed! Final accuracy: {accuracy:.4f}")
    print("The enhanced model should provide much better confidence scores!")
