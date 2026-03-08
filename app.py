import os
# Suppress standard TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 3 = FATAL only
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Turn off oneDNN warnings

# Suppress absl logging (often used by TF internally)
import logging
logging.getLogger('absl').setLevel(logging.ERROR)

import time
import io
import json
import datetime
from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_PATH = 'best_custom_cnn_malaria_model.h5'
model = None
target_size = (130, 130)  # Default fallback

# Global mock stats
scans_today = 2852

print("Loading model...")
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
    # Attempt to infer input shape
    try:
        input_shape = model.input_shape
        if input_shape and len(input_shape) >= 3:
            target_size = (input_shape[1], input_shape[2])
            print(f"Inferred target size: {target_size}")
    except Exception as e:
        print(f"Could not infer input shape: {e}")
except Exception as e:
    print(f"Error loading model: {e}")


def preprocess_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size)
    
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Commonly they're scaled 0-1 via ImageDataGenerator(rescale=1./255)
    img_array = img_array / 255.0
    return img_array


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    global scans_today
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if not model:
        # For testing UI even without model, we fallback to a dummy if needed, but let's send error
        return jsonify({'error': 'Model not loaded on server'}), 500

    try:
        start_time = time.time()
        
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        
        processed_image = preprocess_image(img, target_size)
        prediction = model.predict(processed_image)
        
        # Decode prediction
        pred_shape = prediction.shape
        if len(pred_shape) == 2 and pred_shape[1] > 1:
            # Categorical
            confidence = float(np.max(prediction[0]))
            class_idx = int(np.argmax(prediction[0]))
        else:
            # Binary sigmoid
            prob = float(prediction[0][0])
            if prob > 0.5:
                # likely Uninfected
                class_idx = 1
                confidence = prob
            else:
                # likely Parasitized
                class_idx = 0
                confidence = 1.0 - prob
                
        # 1 = Uninfected, 0 = Parasitized
        result_text = "UNINFECTED" if class_idx == 1 else "PARASITIZED"
        sub_text = "No malaria parasites detected" if class_idx == 1 else "Malaria parasites detected in sample"
        
        processing_time = round(time.time() - start_time, 2)
        if processing_time < 0.01:
            processing_time = 0.01 # ensure non-zero for display
            
        scans_today += 1
        timestamp = datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
        
        return jsonify({
            'success': True,
            'result': result_text,
            'sub_text': sub_text,
            'confidence': confidence,
            'confidence_percent': f"{round(confidence * 100, 2)}%",
            'processing_time': f"{processing_time}s",
            'timestamp': timestamp,
            'scans_today': scans_today
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/stats', methods=['GET'])
def stats():
    return jsonify({
        'scans_today': scans_today
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
