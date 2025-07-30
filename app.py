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
from flask import Flask, request, render_template, jsonify, send_file, redirect, url_for
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

app = Flask(__name__)
CORS(app)  # Enable CORS for API access

# Configure upload folder
UPLOAD_FOLDER = 'static/audio_uploads'
VISUALIZATIONS_FOLDER = 'static/visualizations'
HISTORY_FILE = 'data/emotion_history.csv'
ALLOWED_EXTENSIONS = {'wav', 'mp3'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['VISUALIZATIONS_FOLDER'] = VISUALIZATIONS_FOLDER

# Create necessary folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VISUALIZATIONS_FOLDER, exist_ok=True)
os.makedirs('data', exist_ok=True)

# Load the trained model and scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'emotion_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Emotion mapping with emojis and colors
EMOTION_CONFIG = {
    'happy': {'emoji': '😄', 'color': '#28a745', 'bg_color': '#d4edda'},
    'sad': {'emoji': '😢', 'color': '#6c757d', 'bg_color': '#e2e3e5'},
    'angry': {'emoji': '😠', 'color': '#dc3545', 'bg_color': '#f8d7da'},
    'surprised': {'emoji': '😮', 'color': '#ffc107', 'bg_color': '#fff3cd'},
    'fear': {'emoji': '😨', 'color': '#6f42c1', 'bg_color': '#e2d9f3'},
    'disgust': {'emoji': '🤢', 'color': '#fd7e14', 'bg_color': '#ffeaa7'},
    'neutral': {'emoji': '😐', 'color': '#17a2b8', 'bg_color': '#d1ecf1'}
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(file_path):
    """Extract MFCC features from audio file."""
    try:
        # Load audio file
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        
        # Take mean of MFCCs
        mfccs_mean = np.mean(mfccs.T, axis=0)
        
        return mfccs_mean, audio, sample_rate
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
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
    """Save emotion prediction to history CSV."""
    try:
        # Create history dataframe
        history_data = {
            'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'filename': [filename],
            'predicted_emotion': [emotion],
            'confidence': [max(probabilities.values())],
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
        
        return True
    except Exception as e:
        print(f"Error saving history: {str(e)}")
        return False

def generate_pdf_report(emotion, probabilities, filename, audio_path, viz_path):
    """Generate PDF report with emotion analysis results."""
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
            alignment=1  # Center alignment
        )
        story.append(Paragraph("SentiSound - Audio Emotion Analysis Report", title_style))
        story.append(Spacer(1, 20))
        
        # Basic info
        story.append(Paragraph(f"<b>Audio File:</b> {filename}", styles['Normal']))
        story.append(Paragraph(f"<b>Analysis Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Emotion result
        emotion_config = EMOTION_CONFIG.get(emotion, EMOTION_CONFIG['neutral'])
        story.append(Paragraph(f"<b>Detected Emotion:</b> {emotion_config['emoji']} {emotion.title()}", styles['Heading2']))
        story.append(Spacer(1, 20))
        
        # Probabilities table
        story.append(Paragraph("<b>Emotion Probabilities:</b>", styles['Heading3']))
        prob_data = [['Emotion', 'Probability (%)']]
        for emotion_name, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
            prob_data.append([emotion_name.title(), f"{prob*100:.1f}%"])
        
        prob_table = Table(prob_data, colWidths=[2*inch, 1.5*inch])
        prob_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(prob_table)
        story.append(Spacer(1, 20))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        
        return buffer
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        return None

def get_emotion_suggestions(emotion):
    """Get personalized suggestions based on detected emotion."""
    suggestions = {
        'happy': {
            'music': 'https://www.youtube.com/results?search_query=happy+upbeat+music',
            'activity': 'Share your positive mood with friends and family!',
            'meditation': 'Try gratitude meditation to enhance your happiness.'
        },
        'sad': {
            'music': 'https://www.youtube.com/results?search_query=uplifting+inspirational+music',
            'activity': 'Consider talking to a friend or engaging in a favorite hobby.',
            'meditation': 'Try self-compassion meditation to lift your spirits.'
        },
        'angry': {
            'music': 'https://www.youtube.com/results?search_query=calming+meditation+music',
            'activity': 'Take deep breaths and try some physical exercise.',
            'meditation': 'Practice anger management meditation techniques.'
        },
        'surprised': {
            'music': 'https://www.youtube.com/results?search_query=exciting+adventure+music',
            'activity': 'Channel your surprise into creative activities!',
            'meditation': 'Try mindfulness meditation to process the surprise.'
        },
        'fear': {
            'music': 'https://www.youtube.com/results?search_query=soothing+calm+music',
            'activity': 'Practice grounding techniques and reach out for support.',
            'meditation': 'Try anxiety-reducing meditation practices.'
        },
        'disgust': {
            'music': 'https://www.youtube.com/results?search_query=refreshing+clean+music',
            'activity': 'Engage in activities that bring you joy and comfort.',
            'meditation': 'Practice cleansing meditation techniques.'
        },
        'neutral': {
            'music': 'https://www.youtube.com/results?search_query=peaceful+ambient+music',
            'activity': 'Perfect time for reflection and planning.',
            'meditation': 'Try balanced meditation for inner peace.'
        }
    }
    return suggestions.get(emotion, suggestions['neutral'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
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
        features_scaled = scaler.transform(features.reshape(1, -1))
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        emotion_probs = dict(zip(model.classes_, probabilities))
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
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
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
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
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
def get_history():
    """Get emotion prediction history."""
    try:
        if os.path.exists(HISTORY_FILE):
            df = pd.read_csv(HISTORY_FILE)
            # Convert to list of dictionaries for JSON
            history = df.to_dict('records')
            return jsonify({'history': history})
        else:
            return jsonify({'history': []})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download-report/<filename>')
def download_report(filename):
    """Download PDF report for a specific audio file."""
    try:
        # Get the latest prediction for this file
        if os.path.exists(HISTORY_FILE):
            df = pd.read_csv(HISTORY_FILE)
            file_history = df[df['filename'] == filename]
            if not file_history.empty:
                latest = file_history.iloc[-1]
                
                # Reconstruct probabilities
                probabilities = json.loads(latest['top_3_probabilities'])
                
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
                        download_name=f'sentisound_report_{filename}.pdf',
                        mimetype='application/pdf'
                    )
        
        return jsonify({'error': 'Report not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/record', methods=['POST'])
def handle_recording():
    """Handle real-time voice recording."""
    try:
        # Get base64 audio data
        data = request.get_json()
        audio_data = data.get('audio')
        
        if not audio_data:
            return jsonify({'error': 'No audio data received'}), 400
        
        # Decode base64 audio
        audio_bytes = base64.b64decode(audio_data.split(',')[1])
        
        # Save temporary file
        filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        with open(filepath, 'wb') as f:
            f.write(audio_bytes)
        
        # Process the recording
        features, audio, sample_rate = extract_features(filepath)
        if features is None:
            return jsonify({'error': 'Error processing audio recording'}), 400
        
        # Scale features
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
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
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🎤 Starting SentiSound - Advanced Audio Emotion Detection System")
    print("🌐 Access the application at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000) 