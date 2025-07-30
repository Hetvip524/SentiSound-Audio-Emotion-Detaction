# 🎤 SentiSound - Audio Emotion Detection Backend API

A robust Flask-based backend API for audio emotion recognition using machine learning. This backend provides endpoints for audio upload, emotion prediction, prediction history, PDF report generation, real-time (base64) prediction, and audio visualizations. No frontend/UI/UX code is included—ideal for integration with any client or for backend-focused projects.

---

## 🚀 Features

1. **Audio Upload & Emotion Prediction API**  
   - Upload audio files (.wav, .mp3, .m4a) and receive predicted emotion and probabilities.
2. **Prediction History API**  
   - Stores and retrieves a history of predictions (timestamp, filename, emotion, confidence, top 3 probabilities).
3. **PDF Report Generation API**  
   - Generates downloadable PDF reports for each prediction (emotion, probabilities, timestamp).
4. **Real-time Recording API**  
   - Accepts base64-encoded audio for real-time emotion prediction (testable via Postman/curl).
5. **Visualization Generation**  
   - Generates and saves waveform/spectrogram/MFCC images for each audio file.
6. **Model Info & Health Endpoints**  
   - `/health` for API health checks, `/models/info` for model metadata.

---

## 👥 Work Distribution (for Two-Person Team)

- **Person 1: Model & Feature Engineering**
  - Data collection and preprocessing (RAVDESS dataset).
  - Feature extraction (MFCCs).
  - Model selection, training, evaluation (Random Forest).
  - Saving/loading model and scaler.
  - Writing the training script and documentation.

- **Person 2: Backend API & Data Management**
  - Implementing Flask API endpoints for upload, prediction, history, and PDF report.
  - File handling and validation.
  - Storing and retrieving prediction history.
  - Generating PDF reports and visualizations.
  - Writing API documentation and usage examples.

---

## 📁 Project Structure

```
Audio_Emo_Backend/
├── app.py                    # Main Flask backend API
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── data/
│   └── emotion_history.csv   # Emotion analysis history
├── models/
│   ├── emotion_model.pkl     # Trained ML model
│   └── scaler.pkl            # Feature scaler
├── static/
│   ├── audio_uploads/        # Uploaded audio files
│   └── visualizations/       # Generated audio visualizations
└── train_model.py            # Model training script
```

---

## 🔧 API Documentation

### 1. Audio Prediction Endpoint
```
POST /predict
Content-Type: multipart/form-data
file: <audio file>
```
**Response:**
```
{
  "emotion": "happy",
  "probabilities": {"happy": 0.85, "sad": 0.10, "angry": 0.05},
  "audio_file": "uploaded_file.wav",
  "visualization": "visualizations/uploaded_file_analysis.png"
}
```

### 2. Real-time (Base64) Prediction Endpoint
```
POST /record
Content-Type: application/json
{
  "audio": "data:audio/wav;base64,UklGR..."
}
```
**Response:** Same as above.

### 3. Prediction History
```
GET /history
```
**Response:**
```
{
  "history": [
    {
      "timestamp": "2024-01-15 14:30:25",
      "filename": "recording_20240115_143025.wav",
      "predicted_emotion": "happy",
      "confidence": 0.85,
      "top_3_probabilities": "{\"happy\": 0.85, \"sad\": 0.10, \"angry\": 0.05}"
    }
  ]
}
```

### 4. PDF Report Download
```
GET /download-report/{filename}
```
**Response:** PDF file download

### 5. Model Info
```
GET /models/info
```
**Response:**
```
{
  "model_type": "RandomForestClassifier",
  "classes": ["happy", "sad", ...],
  "n_features": 40,
  "trained_on": "RAVDESS",
  "feature_type": "MFCC (mean, 40 dims)",
  "scaler": "StandardScaler"
}
```

### 6. Health Check
```
GET /health
```
**Response:** `{ "status": "ok" }`

---

## 🧪 Example API Calls

**Audio Prediction (curl):**
```
curl -X POST -F "file=@audio.wav" http://localhost:5000/predict
```

**Real-time (Base64) Prediction (curl):**
```
curl -X POST -H "Content-Type: application/json" -d '{"audio": "data:audio/wav;base64,<BASE64_STRING>"}' http://localhost:5000/record
```

**Get History:**
```
curl http://localhost:5000/history
```

**Download PDF Report:**
```
curl -O http://localhost:5000/download-report/your_audio.wav
```

**Model Info:**
```
curl http://localhost:5000/models/info
```

**Health Check:**
```
curl http://localhost:5000/health
```

---

## 🛠️ How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the backend API:**
   ```bash
   python app.py
   ```
3. **Test endpoints using curl/Postman.**

---

## 📈 Model & Algorithm
- **Feature Extraction:** MFCC (Mel-frequency cepstral coefficients, mean, 40 dims)
- **Model:** Random Forest (scikit-learn)
- **Preprocessing:** StandardScaler
- **Dataset:** RAVDESS
- **Alternatives:** SVM, Neural Networks (can be swapped in train_model.py)

---

## 🧩 Extending the Backend
- Add new endpoints for analytics, user management, or batch processing.
- Swap in new models by retraining and updating the .pkl files.
- Integrate with any frontend or mobile app via REST API.
- Add authentication (Flask-JWT, OAuth) for production use.

---

## 📄 License
MIT License. See LICENSE file.

---

**Made for backend-focused audio emotion recognition projects.**