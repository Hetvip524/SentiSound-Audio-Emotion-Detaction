import os
import json
import base64
import io
import numpy as np
import pandas as pd
import librosa
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from flask import Flask, request, render_template, jsonify, send_file, redirect, url_for, session
from flask_cors import CORS
from werkzeug.utils import secure_filename
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from authlib.integrations.flask_client import OAuth
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)  # Enable CORS for API access

# Configure upload folder
UPLOAD_FOLDER = 'static/audio_uploads'
VISUALIZATIONS_FOLDER = 'static/visualizations'
HISTORY_FILE = 'data/emotion_history.csv'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['VISUALIZATIONS_FOLDER'] = VISUALIZATIONS_FOLDER
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev_secret_change_me')
# Ensure absolute SQLite path to avoid 'unable to open database file'
BASE_PATH_FOR_DB = os.path.dirname(os.path.abspath(__file__))
# Load environment variables from .env in project root (optional)
load_dotenv(os.path.join(BASE_PATH_FOR_DB, '.env'))
ABS_DATA_DIR = os.path.join(BASE_PATH_FOR_DB, 'data')
os.makedirs(ABS_DATA_DIR, exist_ok=True)
ABS_DB_PATH = os.path.join(ABS_DATA_DIR, 'app.db')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', f"sqlite:///{ABS_DB_PATH}")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

login_manager = LoginManager(app)
login_manager.login_view = 'login'

oauth = OAuth(app)
# Prefer environment variables, else try instance/google_oauth.json for persistence
app.config['GOOGLE_CLIENT_ID'] = os.environ.get('GOOGLE_CLIENT_ID', '')
app.config['GOOGLE_CLIENT_SECRET'] = os.environ.get('GOOGLE_CLIENT_SECRET', '')

if (not app.config['GOOGLE_CLIENT_ID'] or not app.config['GOOGLE_CLIENT_SECRET']):
    try:
        instance_dir = os.path.join(BASE_PATH_FOR_DB, 'instance')
        cred_path = os.path.join(instance_dir, 'google_oauth.json')
        if os.path.exists(cred_path):
            with open(cred_path, 'r', encoding='utf-8') as f:
                cred = json.load(f)
            # Support Google-style web client file or flat keys
            if 'web' in cred:
                app.config['GOOGLE_CLIENT_ID'] = cred['web'].get('client_id', '')
                app.config['GOOGLE_CLIENT_SECRET'] = cred['web'].get('client_secret', '')
            else:
                app.config['GOOGLE_CLIENT_ID'] = cred.get('client_id', '')
                app.config['GOOGLE_CLIENT_SECRET'] = cred.get('client_secret', '')
    except Exception as _e:
        pass
if app.config['GOOGLE_CLIENT_ID'] and app.config['GOOGLE_CLIENT_SECRET']:
    oauth.register(
        name='google',
        client_id=app.config['GOOGLE_CLIENT_ID'],
        client_secret=app.config['GOOGLE_CLIENT_SECRET'],
        access_token_url='https://oauth2.googleapis.com/token',
        authorize_url='https://accounts.google.com/o/oauth2/v2/auth',
        client_kwargs={'scope': 'openid email profile'},
        server_metadata_url='https://accounts.google.com/.well-known/openid-configuration'
    )

# Create necessary folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VISUALIZATIONS_FOLDER, exist_ok=True)
os.makedirs('data', exist_ok=True)

# User model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=True)
    name = db.Column(db.String(255), nullable=True)
    provider = db.Column(db.String(50), default='local')

    def __repr__(self):
        return f"<User {self.email}>"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Initialize DB file
db_path = os.path.join('data', 'app.db')
os.makedirs('data', exist_ok=True)
if not os.path.exists(db_path):
    with app.app_context():
        db.create_all()

# Load the trained model, scaler, and feature selector
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Ensure absolute history path
HISTORY_FILE = os.path.join(BASE_DIR, 'data', 'emotion_history.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'emotion_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
FEATURE_SELECTOR_PATH = os.path.join(BASE_DIR, 'models', 'feature_selector.pkl')

# Load model with compatibility shims
model = joblib.load(MODEL_PATH)

# Compatibility shim for scikit-learn version differences
def _apply_sklearn_compat_shims(loaded_model):
    try:
        # Handle RandomForest/ExtraTrees style models with tree estimators
        estimators = getattr(loaded_model, 'estimators_', None)
        if estimators:
            for est in estimators:
                if not hasattr(est, 'monotonic_cst'):
                    setattr(est, 'monotonic_cst', None)
    except Exception as _e:
        pass

_apply_sklearn_compat_shims(model)

scaler = joblib.load(SCALER_PATH)

# Load feature selector if it exists (for enhanced model)
feature_selector = None
if os.path.exists(FEATURE_SELECTOR_PATH):
    try:
        feature_selector = joblib.load(FEATURE_SELECTOR_PATH)
        print(f"Loaded feature selector with {feature_selector.k_} features")
    except Exception as e:
        print(f"Warning: Could not load feature selector: {e}")
        feature_selector = None
else:
    print("No feature selector found - using all features")

# Emotion mapping with emojis and colors
EMOTION_CONFIG = {
    'happy': {'emoji': '😄', 'color': '#28a745', 'bg_color': '#d4edda'},
    'sad': {'emoji': '😢', 'color': '#6c757d', 'bg_color': '#e2e3e5'},
    'angry': {'emoji': '😠', 'color': '#dc3545', 'bg_color': '#f8d7da'},
    'surprised': {'emoji': '😮', 'color': '#ffc107', 'bg_color': '#fff3cd'},
    'fear': {'emoji': '😨', 'color': '#6f42c1', 'bg_color': '#e2d9f3'},
    'disgust': {'emoji': '🤢', 'color': '#fd7e14', 'bg_color': '#ffeaa7'},
    'neutral': {'emoji': ':-|', 'color': '#17a2b8', 'bg_color': '#d1ecf1'}
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(file_path):
    """Extract comprehensive audio features for better emotion recognition."""
    try:
        print(f"Attempting to load audio file: {file_path}")
        
        # Check if file exists and has content
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            return None, None, None
        
        file_size = os.path.getsize(file_path)
        print(f"File size: {file_size} bytes")
        
        if file_size == 0:
            print("File is empty")
            return None, None, None
        
        # Try different audio loading approaches
        try:
            # First attempt: standard librosa load
            audio, sample_rate = librosa.load(file_path, sr=22050, res_type='kaiser_fast')
            print(f"Successfully loaded audio: {len(audio)} samples, {sample_rate} Hz")
        except Exception as e1:
            print(f"Standard librosa load failed: {str(e1)}")
            try:
                # Second attempt: load with different sample rate
                audio, sample_rate = librosa.load(file_path, sr=16000)
                print(f"Loaded with 16kHz: {len(audio)} samples, {sample_rate} Hz")
            except Exception as e2:
                print(f"16kHz load failed: {str(e2)}")
                try:
                    # Third attempt: load without resampling
                    audio, sample_rate = librosa.load(file_path, sr=None)
                    print(f"Loaded without resampling: {len(audio)} samples, {sample_rate} Hz")
                except Exception as e3:
                    print(f"All loading attempts failed: {str(e3)}")
                    return None, None, None
        
        # Check if audio has content
        if len(audio) == 0:
            print("Audio has no samples")
            return None, None, None
        
        # Enhanced preprocessing: trim silence and normalize
        try:
            audio, _ = librosa.effects.trim(audio, top_db=20)
        except Exception:
            pass
        
        audio = librosa.util.normalize(audio)

        # Extract comprehensive features (same as training)
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
        
        # Convert to numpy array and handle any NaN/inf values
        features = np.array(features)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"Successfully extracted {len(features)} comprehensive features")
        return features, audio, sample_rate
        
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None, None, None

def create_visualizations(audio, sample_rate, filename):
    """Create waveform, spectrogram, and MFCC visualizations."""
    try:
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle('SentiSound - Audio Analysis', fontsize=16, fontweight='bold')
        
        # 1. Waveform
        axes[0].plot(np.linspace(0, len(audio)/sample_rate, len(audio)), audio)
        axes[0].set_title('Waveform')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        img = librosa.display.specshow(D, y_axis='linear', x_axis='time', sr=sample_rate, ax=axes[1])
        axes[1].set_title('Spectrogram')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Frequency (Hz)')
        fig.colorbar(img, ax=axes[1], format='%+2.0f dB')
        
        # 3. MFCC
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        img = librosa.display.specshow(mfccs, x_axis='time', sr=sample_rate, ax=axes[2])
        axes[2].set_title('MFCC')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('MFCC Coefficients')
        fig.colorbar(img, ax=axes[2])
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = os.path.join(app.config['VISUALIZATIONS_FOLDER'], f'{filename}_analysis.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return f'visualizations/{filename}_analysis.png'
    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")
        return None

def save_emotion_history(filename, emotion, probabilities, confidence_threshold=0.5):
    """Save emotion prediction to history CSV with all probabilities."""
    try:
        # Create history dataframe
        history_data = {
            'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'filename': [filename],
            'predicted_emotion': [emotion],
            'confidence': [max(probabilities.values())],
            'all_probabilities': [json.dumps(probabilities)],  # Save all probabilities
            'top_3_probabilities': [json.dumps(dict(sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:3]))]
        }
        
        df_new = pd.DataFrame(history_data)
        
        # Load existing history or create new
        if os.path.exists(HISTORY_FILE):
            df_existing = pd.read_csv(HISTORY_FILE)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new
        
        # Save to CSV
        df_combined.to_csv(HISTORY_FILE, index=False)
        print(f"History saved: {filename} -> {HISTORY_FILE}")
        
        return True
    except Exception as e:
        print(f"Error saving history: {str(e)}")
        return False

def generate_pdf_report(emotion, probabilities, filename, audio_path, viz_path):
    """Generate comprehensive PDF report with all emotion analysis details."""
    try:
        # Create PDF buffer
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1,  # Center alignment
            textColor=colors.darkblue
        )
        story.append(Paragraph("🎤 SentiSound - Audio Emotion Analysis Report", title_style))
        story.append(Spacer(1, 20))
        
        # Executive Summary
        story.append(Paragraph("<b>📋 Executive Summary</b>", styles['Heading2']))
        story.append(Spacer(1, 10))
        
        # Basic info
        story.append(Paragraph(f"<b>Audio File:</b> {filename}", styles['Normal']))
        story.append(Paragraph(f"<b>Analysis Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Paragraph(f"<b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Primary Emotion Result
        emotion_config = EMOTION_CONFIG.get(emotion, EMOTION_CONFIG['neutral'])
        story.append(Paragraph(f"<b>🎯 Primary Detected Emotion:</b> {emotion_config['emoji']} {emotion.title()}", styles['Heading2']))
        story.append(Spacer(1, 10))
        
        # Confidence level
        confidence = max(probabilities.values())
        confidence_color = colors.green if confidence > 0.8 else colors.orange if confidence > 0.6 else colors.red
        story.append(Paragraph(f"<b>Confidence Level:</b> <font color='{confidence_color}'>{confidence*100:.1f}%</font>", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Detailed Probabilities Table
        story.append(Paragraph("<b>📊 Detailed Emotion Probabilities:</b>", styles['Heading3']))
        prob_data = [['Rank', 'Emotion', 'Probability (%)', 'Confidence Level']]
        
        sorted_emotions = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        for i, (emotion_name, prob) in enumerate(sorted_emotions, 1):
            confidence_level = "High" if prob > 0.8 else "Medium" if prob > 0.6 else "Low"
            prob_data.append([str(i), emotion_name.title(), f"{prob*100:.1f}%", confidence_level])
        
        prob_table = Table(prob_data, colWidths=[0.5*inch, 1.5*inch, 1.2*inch, 1.2*inch])
        prob_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 10)
        ]))
        story.append(prob_table)
        story.append(Spacer(1, 20))
        
        # Technical Analysis Details
        story.append(Paragraph("<b>🔬 Technical Analysis Details:</b>", styles['Heading3']))
        story.append(Spacer(1, 10))
        
        # Model information
        story.append(Paragraph(f"<b>Model Type:</b> {type(model).__name__}", styles['Normal']))
        story.append(Paragraph(f"<b>Feature Extraction:</b> MFCC (Mel-frequency cepstral coefficients)", styles['Normal']))
        story.append(Paragraph(f"<b>Feature Dimensions:</b> 40", styles['Normal']))
        story.append(Paragraph(f"<b>Training Dataset:</b> RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)", styles['Normal']))
        story.append(Paragraph(f"<b>Preprocessing:</b> StandardScaler normalization", styles['Normal']))
        
        # Model performance metrics (if available)
        try:
            if hasattr(model, 'score'):
                story.append(Paragraph(f"<b>Model Score Method:</b> Available", styles['Normal']))
            if hasattr(model, 'classes_'):
                story.append(Paragraph(f"<b>Supported Emotions:</b> {', '.join(model.classes_)}", styles['Normal']))
            if hasattr(model, 'n_features_in_'):
                story.append(Paragraph(f"<b>Input Features:</b> {model.n_features_in_}", styles['Normal']))
        except Exception as e:
            story.append(Paragraph(f"<b>Model Details:</b> Additional metrics not available", styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # Personalized Recommendations
        story.append(Paragraph("<b>💡 Personalized Recommendations:</b>", styles['Heading3']))
        story.append(Spacer(1, 10))
        
        suggestions = get_emotion_suggestions(emotion)
        story.append(Paragraph(f"<b>🎵 Music Suggestions:</b> {suggestions['music']}", styles['Normal']))
        story.append(Paragraph(f"<b>🏃 Activity Suggestions:</b> {suggestions['activity']}", styles['Normal']))
        story.append(Paragraph(f"<b>🧘 Meditation Suggestions:</b> {suggestions['meditation']}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Audio File Information
        story.append(Paragraph("<b>📁 Audio File Information:</b>", styles['Heading3']))
        story.append(Spacer(1, 10))
        
        if os.path.exists(audio_path):
            file_size = os.path.getsize(audio_path)
            file_size_mb = file_size / (1024 * 1024)
            story.append(Paragraph(f"<b>File Size:</b> {file_size_mb:.2f} MB", styles['Normal']))
            story.append(Paragraph(f"<b>File Path:</b> {audio_path}", styles['Normal']))
            
            # Try to get audio duration and sample rate if possible
            try:
                import librosa
                audio, sr = librosa.load(audio_path, sr=None)
                duration = len(audio) / sr
                story.append(Paragraph(f"<b>Duration:</b> {duration:.2f} seconds", styles['Normal']))
                story.append(Paragraph(f"<b>Sample Rate:</b> {sr} Hz", styles['Normal']))
                story.append(Paragraph(f"<b>Audio Length:</b> {len(audio)} samples", styles['Normal']))
            except Exception as e:
                story.append(Paragraph(f"<b>Audio Details:</b> Could not extract audio metadata", styles['Normal']))
        else:
            story.append(Paragraph("<b>File Status:</b> Audio file not found", styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # Visualization Information
        story.append(Paragraph("<b>📈 Generated Visualizations:</b>", styles['Heading3']))
        story.append(Spacer(1, 10))
        
        if viz_path and os.path.exists(viz_path):
            try:
                # Try to include the actual image in the PDF
                from reportlab.platypus import Image
                img = Image(viz_path, width=4*inch, height=3*inch)
                story.append(img)
                story.append(Spacer(1, 10))
                story.append(Paragraph("Includes: Waveform, Spectrogram, and MFCC visualizations", styles['Normal']))
            except Exception as e:
                # Fallback to text description if image inclusion fails
                story.append(Paragraph(f"<b>Analysis Charts:</b> {viz_path}", styles['Normal']))
                story.append(Paragraph("Includes: Waveform, Spectrogram, and MFCC visualizations", styles['Normal']))
                story.append(Paragraph(f"<i>Note: Image could not be embedded due to: {str(e)}</i>", styles['Normal']))
        else:
            story.append(Paragraph("<b>Analysis Charts:</b> No visualizations generated", styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # Analysis Notes
        story.append(Paragraph("<b>📝 Analysis Notes:</b>", styles['Heading3']))
        story.append(Spacer(1, 10))
        
        if confidence > 0.8:
            story.append(Paragraph("✅ <b>High Confidence:</b> The model is very confident in this emotion prediction.", styles['Normal']))
        elif confidence > 0.6:
            story.append(Paragraph("⚠️ <b>Medium Confidence:</b> The model is moderately confident. Consider the second-highest emotion as well.", styles['Normal']))
        else:
            story.append(Paragraph("❌ <b>Low Confidence:</b> The model has low confidence. The prediction may not be reliable.", styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # Footer
        story.append(Paragraph("<b>Generated by SentiSound - Audio Emotion Detection System</b>", styles['Normal']))
        story.append(Paragraph(f"Report ID: {filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}", styles['Normal']))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        
        return buffer
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        return None

def get_emotion_suggestions(emotion):
    """Get personalized suggestions with links based on detected emotion."""
    suggestions = {
        'happy': {
            'music': 'Happy Hits – Upbeat Pop Mix',
            'music_url': 'https://www.youtube.com/playlist?list=PLMC9KNkIncKseYxDN2niH6glGRWKsLtde',
            'activity': 'Quick energizing workout (10–15 min)',
            'activity_url': 'https://www.youtube.com/watch?v=UBMk30rjy0o',
            'meditation': 'Gratitude practice (5–10 min)',
            'meditation_url': 'https://www.youtube.com/watch?v=WPPPFqsECz0',
            'book': 'The Happiness Advantage – Shawn Achor'
        },
        'sad': {
            'music': 'Cheerful Uplifting Songs',
            'music_url': 'https://www.youtube.com/playlist?list=PLgG8bm3T8oE2K8w2oYj2vuw9S1qQxVwQh',
            'activity': 'Go for a 15‑minute walk or call a friend',
            'activity_url': 'https://www.youtube.com/watch?v=I50ZAs2bU9I',
            'meditation': 'Self‑compassion meditation (10 min)',
            'meditation_url': 'https://www.youtube.com/watch?v=Z9A3QnH8V7c',
            'book': 'Feeling Good – David D. Burns'
        },
        'angry': {
            'music': 'Calming Ambient Sounds',
            'music_url': 'https://www.youtube.com/watch?v=1ZYbU82GVz4',
            'activity': 'Box breathing 4‑4‑4‑4 (5 min) + journaling',
            'activity_url': 'https://www.youtube.com/watch?v=tEmt1Znux58',
            'meditation': 'Anger release (10 min)',
            'meditation_url': 'https://www.youtube.com/watch?v=aGx4IlppSgU',
            'book': 'The Dance of Anger – Harriet Lerner'
        },
        'fearful': {
            'music': 'Soft Piano for Anxiety Relief',
            'music_url': 'https://www.youtube.com/watch?v=KTo_6Q5wGgc',
            'activity': 'Grounding 5‑4‑3‑2‑1 (3–5 min)',
            'activity_url': 'https://www.youtube.com/watch?v=Fs8Gk9WnJvk',
            'meditation': 'Anxiety relief (10 min)',
            'meditation_url': 'https://www.youtube.com/watch?v=O-6f5wQXSu8',
            'book': 'Feel the Fear and Do It Anyway – Susan Jeffers'
        },
        'surprised': {
            'music': 'Feel‑good Jazz Vibes',
            'music_url': 'https://www.youtube.com/playlist?list=PL1Z0x4QJQJ4o9Zz3e1l2QJld7Yv9gk3Zc',
            'activity': 'Channel energy into a creative sketch (10–15 min)',
            'activity_url': 'https://www.youtube.com/watch?v=qZ2rYk4LLxU',
            'meditation': 'Open awareness (5–10 min)',
            'meditation_url': 'https://www.youtube.com/watch?v=Yc5TlrC_z5E',
            'book': 'Range – David Epstein'
        },
        'disgust': {
            'music': 'Nature Ambience (Forest/Water)',
            'music_url': 'https://www.youtube.com/watch?v=tq3m-r1qU8s',
            'activity': 'Declutter a small area (10–15 min)',
            'activity_url': 'https://www.youtube.com/watch?v=4xgZ-q1AO5I',
            'meditation': 'Cleansing breath practice (5 min)',
            'meditation_url': 'https://www.youtube.com/watch?v=RZ3S9XQjcxE',
            'book': 'Atomic Habits – James Clear'
        },
        'calm': {
            'music': 'Lo‑fi Focus Beats (Live)',
            'music_url': 'https://www.youtube.com/watch?v=jfKfPfyJRdk',
            'activity': 'Gentle yoga flow (10–15 min) or journaling',
            'activity_url': 'https://www.youtube.com/watch?v=v7AYKMP6rOE',
            'meditation': 'Mindfulness of breath (10 min)',
            'meditation_url': 'https://www.youtube.com/watch?v=inpok4MKVLM',
            'book': 'The Miracle of Mindfulness – Thich Nhat Hanh'
        },
        'neutral': {
            'music': 'Instrumental Focus (Lo‑fi/Chillhop)',
            'music_url': 'https://www.youtube.com/watch?v=5qap5aO4i9A',
            'activity': 'Plan your next goal (10 min)',
            'activity_url': 'https://www.youtube.com/watch?v=H14bBuluwB8',
            'meditation': '5‑minute reset',
            'meditation_url': 'https://www.youtube.com/watch?v=inpok4MKVLM',
            'book': 'Deep Work – Cal Newport'
        }
    }
    return suggestions.get(emotion, suggestions['neutral'])

def safe_parse_probabilities(json_str):
    """Safely parse JSON string for emotion probabilities."""
    if not json_str or pd.isna(json_str):
        return None
    
    try:
        # Handle double-quoted JSON strings (common CSV issue)
        if isinstance(json_str, str):
            # Remove outer quotes if they exist
            cleaned = json_str.strip()
            if cleaned.startswith('"') and cleaned.endswith('"'):
                cleaned = cleaned[1:-1]
            # Replace non-JSON tokens
            cleaned = cleaned.replace('NaN', 'null').replace("'", '"')
            # Some CSVs may contain trailing commas
            cleaned = cleaned.replace(',}', '}').replace(',]', ']')
            
            # Parse the cleaned JSON
            return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        print(f"Failed to parse probabilities: {e}")
        return None
    
    return None

def sanitize_for_json(obj):
    import math as _m
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and (_m.isnan(obj) or _m.isinf(obj)):
        return None
    return obj

@app.route('/')
def root():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    return render_template('landing.html')

@app.route('/home')
@login_required
def home():
    return render_template('home.html', user=current_user)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.form
        email = (data.get('email') or '').strip().lower()
        password = data.get('password') or ''
        if not email or not password:
            return render_template('login.html', error='Email and password required')
        user = User.query.filter_by(email=email).first()
        if not user or not user.password_hash:
            return render_template('login.html', error='Invalid credentials')
        # Simple hash check using werkzeug.security
        from werkzeug.security import check_password_hash
        if not check_password_hash(user.password_hash, password):
            return render_template('login.html', error='Invalid credentials')
        login_user(user)
        return redirect(url_for('home'))
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        data = request.form
        name = (data.get('name') or '').strip()
        email = (data.get('email') or '').strip().lower()
        password = data.get('password') or ''
        confirm = data.get('confirm_password') or ''

        # Password criteria: >=8 chars, upper, lower, digit
        import re
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
            return render_template('signup.html', error='Enter a valid email', name=name, email=email)
        if password != confirm:
            return render_template('signup.html', error='Passwords do not match', name=name, email=email)
        if len(password) < 8 or not re.search(r"[A-Z]", password) or not re.search(r"[a-z]", password) or not re.search(r"\d", password):
            return render_template('signup.html', error='Password must be 8+ chars with upper, lower, and digit', name=name, email=email)
        if User.query.filter_by(email=email).first():
            return render_template('signup.html', error='Email already registered', name=name, email=email)
        from werkzeug.security import generate_password_hash
        user = User(email=email, name=name, password_hash=generate_password_hash(password))
        db.session.add(user)
        db.session.commit()
        login_user(user)
        return redirect(url_for('home'))
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()
    return redirect(url_for('login'))

@app.route('/login/google')
def login_google():
    if 'google' not in oauth._registry:
        return redirect(url_for('login'))
    redirect_uri = url_for('auth_google_callback', _external=True)
    return oauth.google.authorize_redirect(redirect_uri)

@app.route('/auth/google/callback')
def auth_google_callback():
    if 'google' not in oauth._registry:
        return redirect(url_for('login'))
    token = oauth.google.authorize_access_token()
    userinfo = token.get('userinfo') or oauth.google.parse_id_token(token)
    if not userinfo:
        return redirect(url_for('login'))
    email = (userinfo.get('email') or '').lower()
    name = userinfo.get('name')
    user = User.query.filter_by(email=email).first()
    if not user:
        user = User(email=email, name=name, provider='google')
        db.session.add(user)
        db.session.commit()
    login_user(user)
    return redirect(url_for('home'))

@app.route('/analyze')
@login_required
def analyze_page():
    return render_template('index.html')

@app.route('/history-page')
@login_required
def history_page():
    return render_template('history.html')

@app.route('/account')
@login_required
def account():
    # basic account info page
    return render_template('account.html', user=current_user)

@app.route('/account', methods=['POST'])
@login_required
def account_update():
    name = (request.form.get('name') or '').strip()
    phone = (request.form.get('phone') or '').strip()
    avatar = request.files.get('avatar')
    if name:
        current_user.name = name
    if phone:
        current_user.phone = phone
    # Save avatar
    if avatar and avatar.filename:
        os.makedirs(os.path.join(BASE_DIR, 'static', 'avatars'), exist_ok=True)
        fname = secure_filename(f"user_{current_user.id}_" + avatar.filename)
        apath = os.path.join(BASE_DIR, 'static', 'avatars', fname)
        avatar.save(apath)
        current_user.avatar_path = f"/static/avatars/{fname}"
    db.session.commit()
    return redirect(url_for('account'))

@app.route('/audio/list')
@login_required
def audio_list():
    try:
        folder = os.path.join(BASE_DIR, UPLOAD_FOLDER.replace('static/', 'static/'))
        paths = []
        for f in os.listdir(folder):
            if allowed_file(f):
                full = os.path.join(folder, f)
                paths.append((f, os.path.getmtime(full)))
        paths.sort(key=lambda x: x[1], reverse=True)
        files = [p[0] for p in paths[:50]]
        return jsonify({'files': files})
    except Exception as e:
        return jsonify({'files': [], 'error': str(e)})

@app.route('/result', methods=['POST'])
@login_required
def result():
    if 'file' not in request.files:
        return render_template('index.html', error='No file uploaded')
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No file selected')
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        features, audio, sample_rate = extract_features(filepath)
        if features is None:
            return render_template('index.html', error='Error processing audio file')
        # Scale features
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        # Apply feature selection if available
        if feature_selector is not None:
            features_selected = feature_selector.transform(features_scaled)
        else:
            features_selected = features_scaled
        
        # Make prediction
        prediction = model.predict(features_selected)[0]
        probabilities = model.predict_proba(features_selected)[0]
        import numpy as _np
        probabilities = _np.clip(probabilities, 0.0, 1.0)
        probabilities = probabilities / probabilities.sum() if probabilities.sum() > 0 else _np.ones_like(probabilities) / len(probabilities)
        emotion_probs = dict(zip(model.classes_, probabilities))
        # Save to history for web flow too
        save_emotion_history(filename, prediction, emotion_probs)
        viz_path = create_visualizations(audio, sample_rate, filename)
        pdf_url = url_for('download_report', filename=filename)
        return render_template('index.html', result={
            'emotion': prediction,
            'probabilities': emotion_probs,
            'audio_file': filename,
            'visualization': viz_path,
            'pdf_url': pdf_url
        })
    return render_template('index.html', error='Invalid file type')

@app.route('/health')
def health():
    return jsonify({'status': 'ok'}), 200

@app.route('/models/info')
def model_info():
    info = {
        'model_type': str(type(model).__name__),
        'classes': list(getattr(model, 'classes_', [])),
        'n_features': getattr(model, 'n_features_in_', None),
        'trained_on': 'RAVDESS',
        'feature_type': 'MFCC (mean, 40 dims)',
        'scaler': str(type(scaler).__name__),
    }
    return jsonify(info)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract features
        features, audio, sample_rate = extract_features(filepath)
        if features is None:
            return jsonify({'error': 'Error processing audio file'}), 400
        
        # Scale features
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        # Apply feature selection if available
        if feature_selector is not None:
            features_selected = feature_selector.transform(features_scaled)
        else:
            features_selected = features_scaled
        
        # Make prediction
        prediction = model.predict(features_selected)[0]
        probabilities = model.predict_proba(features_selected)[0]
        # Normalize numeric stability to ensure sum=1 and probs in [0,1]
        import numpy as _np
        probabilities = _np.clip(probabilities, 0.0, 1.0)
        if probabilities.sum() <= 0 or not _np.isfinite(probabilities.sum()):
            probabilities = _np.ones_like(probabilities) / len(probabilities)
        else:
            probabilities = probabilities / probabilities.sum()
        
        # Get emotion probabilities
        emotion_probs = dict(zip(model.classes_, probabilities))
        
        # Create visualizations
        viz_path = create_visualizations(audio, sample_rate, filename)
        
        # Save to history
        save_emotion_history(filename, prediction, emotion_probs)
        
        # Get suggestions
        suggestions = get_emotion_suggestions(prediction)
        
        # Get emotion config
        emotion_config = EMOTION_CONFIG.get(prediction, EMOTION_CONFIG['neutral'])
        
        return jsonify({
            'emotion': prediction,
            'probabilities': emotion_probs,
            'audio_file': filename,
            'visualization': viz_path
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    """API endpoint for external access."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract features
        features, _, _ = extract_features(filepath)
        if features is None:
            return jsonify({'error': 'Error processing audio file'}), 400
        
        # Scale features
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        # Apply feature selection if available
        if feature_selector is not None:
            features_selected = feature_selector.transform(features_scaled)
        else:
            features_selected = features_scaled
        
        # Make prediction
        prediction = model.predict(features_selected)[0]
        probabilities = model.predict_proba(features_selected)[0]
        # Normalize
        import numpy as _np
        probabilities = _np.clip(probabilities, 0.0, 1.0)
        if probabilities.sum() <= 0 or not _np.isfinite(probabilities.sum()):
            probabilities = _np.ones_like(probabilities) / len(probabilities)
        else:
            probabilities = probabilities / probabilities.sum()
        
        # Get emotion probabilities
        emotion_probs = dict(zip(model.classes_, probabilities))
        confidence = max(emotion_probs.values())
        
        return jsonify({
            'emotion': prediction,
            'confidence': round(confidence, 3),
            'probabilities': {k: round(v, 3) for k, v in emotion_probs.items()},
            'filename': filename
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/history')
@login_required
def get_history():
    """Get emotion prediction history with detailed information."""
    try:
        if os.path.exists(HISTORY_FILE):
            df = pd.read_csv(HISTORY_FILE)
            print(f"History loaded: {len(df)} entries")
            
            # Sort by timestamp (newest first)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp', ascending=False)
            
            # Convert to list of dictionaries for JSON
            history = df.to_dict('records')
            
            # Process each history item to include more details
            for idx, item in enumerate(history):
                try:
                    # Parse probabilities if available
                    if 'all_probabilities' in item and item['all_probabilities']:
                        item['probabilities'] = safe_parse_probabilities(item['all_probabilities'])
                    elif 'top_3_probabilities' in item and item['top_3_probabilities']:
                        item['probabilities'] = safe_parse_probabilities(item['top_3_probabilities'])
                    else:
                        item['probabilities'] = {item['predicted_emotion']: item['confidence']}
                    
                    # Add emotion emoji
                    emotion = item['predicted_emotion']
                    item['emoji'] = EMOTION_CONFIG.get(emotion, {}).get('emoji', ':-|')
                    
                    # Format timestamp for display
                    item['formatted_time'] = item['timestamp']
                    
                    # Check if visualization exists
                    viz_path = os.path.join(app.config['VISUALIZATIONS_FOLDER'], f"{item['filename']}_analysis.png")
                    item['has_visualization'] = os.path.exists(viz_path)
                    
                    # Sanitize entire item to ensure valid JSON (replace NaN/Inf with None)
                    history[idx] = sanitize_for_json(item)
                    
                except Exception as parse_error:
                    print(f"Error parsing history item: {parse_error}")
                    item['probabilities'] = {}
                    item['emoji'] = '?'
                    item['has_visualization'] = False
                    history[idx] = sanitize_for_json(item)
            
            return jsonify({'history': history})
        else:
            return jsonify({'history': []})
    except Exception as e:
        print(f"Error getting history: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/history/<filename>')
@login_required
def get_history_item(filename):
    """Get detailed information for a specific history item."""
    try:
        if os.path.exists(HISTORY_FILE):
            df = pd.read_csv(HISTORY_FILE)
            file_history = df[df['filename'] == filename]
            
            if not file_history.empty:
                item = file_history.iloc[-1].to_dict()
                
                # Parse probabilities
                try:
                    if 'all_probabilities' in item and item['all_probabilities']:
                        item['probabilities'] = safe_parse_probabilities(item['all_probabilities'])
                    elif 'top_3_probabilities' in item and item['top_3_probabilities']:
                        item['probabilities'] = safe_parse_probabilities(item['top_3_probabilities'])
                    else:
                        item['probabilities'] = {item['predicted_emotion']: item['confidence']}
                except:
                    item['probabilities'] = {item['predicted_emotion']: item['confidence']}
                
                # Add emotion emoji
                emotion = item['predicted_emotion']
                item['emoji'] = EMOTION_CONFIG.get(emotion, {}).get('emoji', ':-|')
                
                # Check if visualization exists
                viz_path = os.path.join(app.config['VISUALIZATIONS_FOLDER'], f"{filename}_analysis.png")
                item['has_visualization'] = os.path.exists(viz_path)
                item['visualization_path'] = f"visualizations/{filename}_analysis.png" if item['has_visualization'] else None
                
                # Ensure JSON-safe output
                return jsonify(sanitize_for_json(item))
            else:
                return jsonify({'error': 'History item not found'}), 404
        else:
            return jsonify({'error': 'No history available'}), 404
    except Exception as e:
        print(f"Error getting history item: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download-report/<filename>')
@login_required
def download_report(filename):
    """Download comprehensive PDF report for a specific audio file."""
    try:
        # Get the latest prediction for this file
        if os.path.exists(HISTORY_FILE):
            df = pd.read_csv(HISTORY_FILE)
            file_history = df[df['filename'] == filename]
            if not file_history.empty:
                latest = file_history.iloc[-1]
                
                # Reconstruct probabilities - try to get all probabilities if available
                try:
                    if 'all_probabilities' in latest:
                        probabilities = safe_parse_probabilities(latest['all_probabilities'])
                    elif 'top_3_probabilities' in latest:
                        probabilities = safe_parse_probabilities(latest['top_3_probabilities'])
                    else:
                        # If only basic info is stored, create a basic probability dict
                        probabilities = {latest['predicted_emotion']: latest['confidence']}
                    
                    # If parsing failed, use fallback
                    if probabilities is None:
                        probabilities = {latest['predicted_emotion']: latest['confidence']}
                except:
                    # Fallback to basic probability
                    probabilities = {latest['predicted_emotion']: latest['confidence']}
                
                # Generate PDF
                audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                viz_path = os.path.join(app.config['VISUALIZATIONS_FOLDER'], f'{filename}_analysis.png')
                
                pdf_buffer = generate_pdf_report(
                    latest['predicted_emotion'],
                    probabilities,
                    filename,
                    audio_path,
                    viz_path if os.path.exists(viz_path) else None
                )
                
                if pdf_buffer:
                    return send_file(
                        pdf_buffer,
                        as_attachment=True,
                        download_name=f'sentisound_comprehensive_report_{filename}.pdf',
                        mimetype='application/pdf'
                    )
        
        return jsonify({'error': 'Report not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/record', methods=['POST'])
@login_required
def handle_recording():
    """Handle real-time voice recording."""
    try:
        # Get base64 audio data
        data = request.get_json()
        audio_data = data.get('audio')
        
        if not audio_data:
            return jsonify({'error': 'No audio data received'}), 400
        
        try:
            # Decode base64 audio - handle different formats
            if ',' in audio_data:
                # Format: data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEA...
                audio_bytes = base64.b64decode(audio_data.split(',')[1])
            else:
                # Format: UklGRiQAAABXQVZFZm10IBAAAAABAAEA...
                audio_bytes = base64.b64decode(audio_data)
        except Exception as decode_error:
            return jsonify({'error': f'Invalid audio data format: {str(decode_error)}'}), 400
        
        # Save temporary file with better naming
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"recording_{timestamp}.wav"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            with open(filepath, 'wb') as f:
                f.write(audio_bytes)
            print(f"Successfully saved audio file: {filepath}")
        except Exception as file_error:
            print(f"Error saving audio file: {str(file_error)}")
            return jsonify({'error': f'Error saving audio file: {str(file_error)}'}), 500
        
        # Verify file was saved
        if not os.path.exists(filepath):
            return jsonify({'error': 'Audio file was not saved properly'}), 500
        
        file_size = os.path.getsize(filepath)
        print(f"Saved file size: {file_size} bytes")
        
        if file_size == 0:
            return jsonify({'error': 'Audio file is empty - recording may have failed'}), 400
        
        # Process the recording
        print("Starting feature extraction...")
        features, audio, sample_rate = extract_features(filepath)
        
        if features is None:
            # Try to provide more specific error information
            if not os.path.exists(filepath):
                error_msg = "Audio file was not saved"
            elif os.path.getsize(filepath) == 0:
                error_msg = "Audio file is empty - no audio data recorded"
            else:
                error_msg = "Failed to extract audio features - file may be corrupted or in unsupported format"
            
            return jsonify({'error': error_msg}), 400
        
        # Scale features
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        # Apply feature selection if available
        if feature_selector is not None:
            features_selected = feature_selector.transform(features_scaled)
        else:
            features_selected = features_scaled
        
        # Make prediction
        prediction = model.predict(features_selected)[0]
        probabilities = model.predict_proba(features_selected)[0]
        
        # Get emotion probabilities
        emotion_probs = dict(zip(model.classes_, probabilities))
        
        # Create visualizations
        viz_path = create_visualizations(audio, sample_rate, filename)
        
        # Save to history
        save_emotion_history(filename, prediction, emotion_probs)
        
        return jsonify({
            'emotion': prediction,
            'probabilities': emotion_probs,
            'audio_file': filename,
            'visualization': viz_path
        })
        
    except Exception as e:
        print(f"Error in handle_recording: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    try:
        print("🎤 Starting SentiSound - Advanced Audio Emotion Detection System")
        print("🌐 Access the application at: http://localhost:5000")
    except UnicodeEncodeError:
        # Fallback for terminals that can't render emojis (e.g., cp1252 on Windows)
        print("Starting SentiSound - Advanced Audio Emotion Detection System")
        print("Access the application at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)