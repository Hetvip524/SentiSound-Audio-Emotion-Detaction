#!/usr/bin/env python3
"""
Simple test script to check if the main app can run without memory issues
"""

import sys
import os

def test_imports():
    """Test imports one by one to identify the problematic one"""
    print("Testing imports...")
    
    try:
        print("✓ Importing os...")
        import os
        
        print("✓ Importing json...")
        import json
        
        print("✓ Importing base64...")
        import base64
        
        print("✓ Importing io...")
        import io
        
        print("✓ Importing numpy...")
        import numpy as np
        
        print("✓ Importing pandas...")
        import pandas as pd
        
        print("✓ Importing librosa...")
        import librosa
        
        print("✓ Importing joblib...")
        import joblib
        
        print("✓ Importing matplotlib...")
        import matplotlib.pyplot as plt
        
        print("✓ Importing seaborn...")
        import seaborn as sns
        
        print("✓ Importing datetime...")
        from datetime import datetime
        
        print("✓ Importing Flask...")
        from flask import Flask, request, render_template, jsonify, send_file, redirect, url_for
        
        print("✓ Importing CORS...")
        from flask_cors import CORS
        
        print("✓ Importing werkzeug...")
        from werkzeug.utils import secure_filename
        
        print("✓ All imports successful!")
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_model_loading():
    """Test if the model files can be loaded"""
    print("\nTesting model loading...")
    
    try:
        import joblib
        
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_PATH = os.path.join(BASE_DIR, 'models', 'emotion_model.pkl')
        SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
        
        if not os.path.exists(MODEL_PATH):
            print(f"✗ Model file not found: {MODEL_PATH}")
            return False
            
        if not os.path.exists(SCALER_PATH):
            print(f"✗ Scaler file not found: {SCALER_PATH}")
            return False
        
        print("✓ Loading model...")
        model = joblib.load(MODEL_PATH)
        
        print("✓ Loading scaler...")
        scaler = joblib.load(SCALER_PATH)
        
        print("✓ Model and scaler loaded successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False

def test_flask_app():
    """Test if Flask app can be created"""
    print("\nTesting Flask app creation...")
    
    try:
        from flask import Flask
        from flask_cors import CORS
        
        app = Flask(__name__)
        CORS(app)
        
        print("✓ Flask app created successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Flask app creation failed: {e}")
        return False

if __name__ == "__main__":
    print("SentiSound - Application Test")
    print("=" * 40)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Cannot proceed.")
        sys.exit(1)
    
    # Test model loading
    if not test_model_loading():
        print("\n❌ Model loading test failed. Cannot proceed.")
        sys.exit(1)
    
    # Test Flask app
    if not test_flask_app():
        print("\n❌ Flask app test failed. Cannot proceed.")
        sys.exit(1)
    
    print("\n✅ All tests passed! The application should be able to run.")
    print("\nTo start the application, run:")
    print("python app.py") 