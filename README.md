# 🎤  SentiSound - Audio Emotion Recognition System

## 📋 Project Explanation

SentiSound is an audio emotion detection application that analyzes speech recordings to identify the emotional state of the speaker. The project aims to provide accurate emotion recognition from audio inputs, making it useful for applications in mental health monitoring, customer service analysis, and human-computer interaction research.

The application allows users to upload audio files or record audio directly through the interface. It then processes this audio to extract relevant features and predicts the emotional content using machine learning techniques. Results are presented with visualizations and detailed probability breakdowns for each emotion category.

## 🎯  Project Overview

### **Core Objective**
Develop a robust, scalable backend system capable of analyzing audio files and accurately predicting emotional states using state-of-the-art machine learning techniques.

### **Technical Innovation**
- **Advanced Audio Processing**: Implements MFCC (Mel-frequency Cepstral Coefficients) feature extraction for optimal emotion recognition
- **Machine Learning Pipeline**: Utilizes Random Forest classification with comprehensive feature engineering
- **Real-time Processing**: Supports both file upload and base64-encoded audio for immediate analysis
- **Professional Reporting**: Automated PDF report generation with detailed analysis and visualizations

### **Supported Emotions**
- 😄 **Happy** - Positive, joyful emotional states
- 😢 **Sad** - Melancholic, sorrowful expressions
- 😠 **Angry** - Aggressive, frustrated emotions
- 😮 **Surprised** - Shocked, astonished reactions
- 😨 **Fear** - Anxious, frightened states
- 🤢 **Disgust** - Repulsed, averse emotions
- 😐 **Neutral** - Balanced, unemotional states

## 🔧 Technologies Used

SentiSound is built using the following technologies:

1. **Flask**: A lightweight Python web framework that serves as the backbone of the application, handling routing, request processing, and response generation.

2. **SQLAlchemy**: An ORM (Object-Relational Mapping) library used for database interactions, storing user accounts and analysis history.

3. **Librosa**: A Python library for audio analysis that enables feature extraction from audio files, including MFCCs (Mel-frequency cepstral coefficients).

4. **Scikit-learn**: A machine learning library that provides the Random Forest classifier used for emotion prediction.

5. **Matplotlib & Seaborn**: Visualization libraries used to generate waveforms, spectrograms, and other audio visualizations.

6. **ReportLab**: A PDF generation library used to create downloadable reports of emotion analysis results.

7. **Flask-Login & Authlib**: Authentication libraries that handle user account management and Google OAuth integration.

## 🔬 Features

1. **Audio Upload & Emotion Prediction**
   - Upload audio files (.wav, .mp3, .m4a) and receive predicted emotion and probabilities.
2. **Advanced Preprocessing for Accuracy**
   - Spectral-gating noise reduction, silence trim, voice-activity detection (VAD) to keep only voiced segments, optional pre‑emphasis.
3. **Unified Navy Theme**
   - Consistent blue palette across Analyze, Home, Account, Login/Signup, Landing.
4. **Working History**
   - Saves every analysis with timestamp, filename, emotion, confidence, and all probabilities; viewable in the UI.
5. **Personalized Suggestions**
   - Clickable music/activity/meditation links with suggested durations; book recommendations per emotion.
6. **PDF Reports**
   - Download comprehensive reports including visuals and probabilities.
7. **Real‑time Recording**
   - Accepts base64 audio; analyzes in one request.
8. **Visualizations & Metadata**
   - Waveform, spectrogram, MFCC charts; `/models/info` for model metadata; `/health` for health checks.

## 🏗️ How the Technologies Work Together

1. **Audio Processing Pipeline**:
   - Audio files are uploaded or recorded through the Flask web interface
   - Librosa processes the audio to extract MFCC features
   - Spectral-gating noise reduction and voice-activity detection (VAD) clean the audio
   - Features are scaled using StandardScaler from scikit-learn
   - The trained Random Forest model predicts emotion probabilities

2. **User Interface Flow**:
   - Flask routes handle different pages (home, login, analysis, history)
   - Templates render the HTML interface with dynamic content
   - CSS styling provides a consistent navy blue theme
   - JavaScript handles real-time recording and form submissions

3. **Data Storage**:
   - SQLite database (app.db) stores user accounts and analysis history
   - File system stores uploaded audio files and generated visualizations

##  📊 Model Details and Accuracy

SentiSound uses a **Random Forest Classifier** for emotion detection, which was selected after comparative testing against SVM and Neural Network alternatives.

**Feature Extraction**:
- 40-dimensional Mel-frequency cepstral coefficients (MFCCs)
- Features are standardized using scikit-learn's StandardScaler

**Training Dataset**:
- RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- Contains professional actors performing speech with different emotional expressions

**Preprocessing Steps**:
- Spectral-gating noise reduction
- Silence trimming
- Voice-activity detection (VAD)
- Pre-emphasis filtering

**Model Performance**:
- Cross-validated accuracy: ~75% on the RAVDESS dataset
- Emotions detected: Happy, Sad, Angry, Fearful, Disgusted, Neutral, Surprised
- Confusion most commonly occurs between similar emotions (e.g., angry/disgusted)

The model can be retrained using the provided training scripts:
- `train_model.py`: Basic training with cross-validation
- `train_model_enhanced.py`: Enhanced training with additional feature engineering

## 🚀 How to Run the Application

### Prerequisites
- Python 3.10 or higher
- ffmpeg (recommended for better audio processing)

### Installation Steps
1. Clone the repository or download the source code

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```

4. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

### Optional Configuration
- For Google Sign-In, create a file at `instance/google_oauth.json` with your OAuth credentials:
  ```json
  {"web":{"client_id":"YOUR_ID","client_secret":"YOUR_SECRET"}}
  ```

- To retrain the model with your own data or parameters:
  ```bash
  python train_model.py
  ```

### Testing
- Run the test suite to verify functionality:
  ```bash
  python test_app.py
  ```

### Windows Troubleshooting
- If you encounter Unicode emoji display issues, the app will automatically fall back to ASCII
- Librosa warnings about 'resampy' are handled internally
- JSON parsing errors with NaN values are sanitized in history responses