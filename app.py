import os

# Force TensorFlow to use the CPU. This is the most stable configuration.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from flask import Flask, render_template, jsonify, request
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json

app = Flask(__name__)

# --- Configuration ---
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.6
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'waste_classifier.h5')
LABELS_PATH = os.path.join(BASE_DIR, 'models', 'class_labels.json')

# --- Global Variables ---
model = None
class_labels = None

# --- Model Loading ---
def load_model_and_labels():
    global model, class_labels
    try:
        print("Loading classification model...")
        model = keras.models.load_model(MODEL_PATH)
        print("✓ Model loaded successfully!")
        with open(LABELS_PATH, 'r') as f:
            class_labels = json.load(f)
        print(f"✓ Class labels loaded: {class_labels}")
        return True
    except Exception as e:
        print(f"✗ Error loading model or labels: {e}")
        return False

# --- Core Image Processing Logic ---
def preprocess_image(img_array):
    """Resizes and normalizes the image for the model."""
    img = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def classify_image(img_array):
    """Classifies the image and returns a dictionary with results."""
    if model is None or img_array is None:
        return {}
    
    try:
        preprocessed = preprocess_image(img_array)
        prediction_val = float(model.predict(preprocessed, verbose=0)[0][0])
        
        class_idx = 1 if prediction_val > 0.5 else 0
        confidence = prediction_val if class_idx == 1 else (1 - prediction_val)
        class_name = class_labels[str(class_idx)]
        
        CLASS_INFO = {
            'B': {'full_name': 'Biodegradable', 'color': '#28a745', 'icon': '🌱'},
            'N': {'full_name': 'Non-Biodegradable', 'color': '#dc3545', 'icon': '♻️'}
        }
        info = CLASS_INFO.get(class_name, {'full_name': 'Unknown', 'color': '#6c757d', 'icon': '❓'})
        
        return {
            'class': str(class_name),
            'confidence': float(confidence),
            'full_name': str(info['full_name']),
            'color': str(info['color']),
            'icon': str(info['icon']),
            'high_confidence': bool(confidence >= CONFIDENCE_THRESHOLD)
        }
    except Exception as e:
        print(f"CRITICAL: Classification error -> {e}")
        return {}

# --- Flask Routes ---
@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_uploaded_image():
    """Handles the image upload and returns the classification result."""
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file provided'}), 400
    
    file = request.files['image']
    try:
        in_memory_file = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(in_memory_file, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'success': False, 'error': 'Invalid image format'}), 400
            
        prediction = classify_image(frame)
        if not prediction:
            return jsonify({'success': False, 'error': 'Classification failed on server.'}), 500
            
        return jsonify({'success': True, 'prediction': prediction})
    except Exception as e:
        print(f"Error processing uploaded image: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Allows the user to shut down the server from the web UI."""
    print("Shutdown request received. Stopping server...")
    shutdown_func = request.environ.get('werkzeug.server.shutdown')
    if shutdown_func:
        shutdown_func()
    else:
        # A fallback for environments where the shutdown function isn't available
        os._exit(0)
    return "Server is shutting down..."

# --- Main Execution ---
if __name__ == '__main__':
    print("\n" + "="*60)
    print("WASTE CLASSIFIER - Image Upload Mode")
    print("="*60 + "\n")
    if load_model_and_labels():
        print("✓ Server ready! Open your browser to http://127.0.0.1:5000")
        app.run(debug=True, threaded=False, host='0.0.0.0', port=5000, use_reloader=False)
    else:
        print("\n✗ CRITICAL: Failed to load model! The application cannot start.")
