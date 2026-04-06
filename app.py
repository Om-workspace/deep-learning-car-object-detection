import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the saved model globally
MODEL_PATH = "models/car_detection_resnet50.h5"
try:
    model = load_model(MODEL_PATH)
    print("AI Model loaded successfully.")
except Exception as e:
    model = None
    print(f"Warning: Model not found or could not be loaded ({e})")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model is not loaded. Train the model first.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess and predict
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'error': 'Invalid image format.'}), 400
            
        image = cv2.resize(image, (224, 224))
        image = np.array(image, dtype=np.float32) / 255.0
        image = np.expand_dims(image, axis=0)

        prediction_val = model.predict(image)[0][0]
        
        # Cleanup uploaded file to save space
        try:
            os.remove(filepath)
        except:
            pass

        confidence = float(prediction_val)
        is_car = confidence > 0.5
        
        # Format the output label
        label = "Car Detected" if is_car else "No Car Detected"
        # If it's a car, confidence is the raw value, else it's 1 - raw value (since the network sigmoid outputs probability of Car)
        display_confidence = confidence if is_car else (1.0 - confidence)

        return jsonify({
            'success': True,
            'label': label,
            'confidence': display_confidence,
            'is_car': is_car
        })

if __name__ == '__main__':
    print("Starting DeepVision AI Flask Server on 0.0.0.0 (Docker friendly)...")
    app.run(host='0.0.0.0', port=5000)
