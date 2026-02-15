from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf
import json
import base64
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for mobile app to connect

# --- CONFIGURATION ---
model_path = 'isl_cnn_model.h5'
label_path = 'model_labels.json'
CONFIDENCE_THRESHOLD = 0.50

# Load Model & Labels at startup
print("Loading Model...")
try:
    model = tf.keras.models.load_model(model_path)
    print("Model Loaded Successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

with open(label_path, 'r') as f:
    labels = json.load(f)
    labels = {int(k): v for k, v in labels.items()}

@app.route('/predict', methods=['POST'])
def predict_sign():
    try:
        # Get base64 image from request
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Remove base64 header if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 to image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to numpy array
        frame = np.array(image)
        
        # Convert RGB to BGR (OpenCV format)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Preprocessing
        roi_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(roi_rgb, (224, 224))
        img = np.expand_dims(img, axis=0)
        img = img / 255.0
        
        # Predict
        pred = model.predict(img, verbose=0)
        class_id = np.argmax(pred)
        confidence = float(np.max(pred))
        
        predicted_char = labels.get(int(class_id), "Unknown")
        
        print(f"Detected: {predicted_char} | Confidence: {confidence:.2f}")
        
        return jsonify({
            'character': predicted_char,
            'confidence': confidence,
            'success': confidence > CONFIDENCE_THRESHOLD
        })
        
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'running',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    # Find your local IP address and run: ipconfig (Windows) or ifconfig (Mac/Linux)
    # Use 0.0.0.0 to make it accessible on local network
    app.run(host='0.0.0.0', port=5000, debug=True)