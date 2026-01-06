"""
Flask Web Application untuk Handwriting Recognition
Interface browser untuk menggambar dan mengenali tulisan tangan
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import base64
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for production

# Load trained model
MODEL_PATH = 'models/best_model.h5'
model = None

def load_trained_model():
    """Load trained model saat aplikasi startup"""
    global model
    if os.path.exists(MODEL_PATH):
        print(f"Loading model dari {MODEL_PATH}...")
        model = load_model(MODEL_PATH)
        print("Model loaded successfully!")
        return True
    else:
        print(f"Warning: Model tidak ditemukan di {MODEL_PATH}")
        print("Silakan train model terlebih dahulu dengan menjalankan: python train.py")
        return False


def preprocess_image(image_data):
    """
    Preprocess gambar dari canvas untuk prediksi
    
    Args:
        image_data: Base64 encoded image dari canvas
    
    Returns:
        Preprocessed image array siap untuk prediksi
    """
    # Decode base64 image
    image_data = image_data.split(',')[1]  # Remove "data:image/png;base64," prefix
    image_bytes = base64.b64decode(image_data)
    
    # Convert ke numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    # Resize ke 28x28
    img = cv2.resize(img, (28, 28))
    
    # Invert colors (canvas adalah hitam di background putih, kita butuh putih di background hitam)
    img = 255 - img
    
    # Normalize ke [0, 1]
    img = img.astype('float32') / 255.0
    
    # Reshape untuk model input
    img = img.reshape(1, 28, 28, 1)
    
    return img


@app.route('/')
def index():
    """Render halaman utama"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint untuk prediksi tulisan tangan
    
    Request JSON:
        {
            "image": "data:image/png;base64,..."
        }
    
    Response JSON:
        {
            "success": true,
            "prediction": "A",
            "confidence": 0.95,
            "probabilities": {...}
        }
    """
    try:
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model belum di-load. Silakan train model terlebih dahulu.'
            }), 500
        
        # Get image data dari request
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({
                'success': False,
                'error': 'Tidak ada data gambar'
            }), 400
        
        # Preprocess image
        processed_image = preprocess_image(image_data)
        
        # Predict
        predictions = model.predict(processed_image, verbose=0)[0]
        
        # Get predicted class dan confidence
        predicted_class = np.argmax(predictions)
        confidence = float(predictions[predicted_class])
        predicted_letter = chr(65 + predicted_class)  # 65 adalah ASCII code untuk 'A'
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions)[-3:][::-1]
        top_3_predictions = {
            chr(65 + i): float(predictions[i]) 
            for i in top_3_indices
        }
        
        return jsonify({
            'success': True,
            'prediction': predicted_letter,
            'confidence': confidence,
            'top_predictions': top_3_predictions
        })
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'model_loaded': model is not None
    })


if __name__ == '__main__':
    print("="*60)
    print("HANDWRITING RECOGNITION - WEB APPLICATION")
    print("="*60)
    
    # Load model
    model_loaded = load_trained_model()
    
    if model_loaded:
        # Get PORT from environment (Railway) or use 5000 for local
        port = int(os.environ.get('PORT', 5000))
        
        print("\nStarting Flask server...")
        print(f"Server akan berjalan di port: {port}")
        print("Tekan Ctrl+C untuk stop server")
        print("="*60)
        
        # Use debug=False for production
        is_production = os.environ.get('RAILWAY_ENVIRONMENT') is not None
        app.run(debug=not is_production, host='0.0.0.0', port=port)
    else:
        print("\nAPLIKASI TIDAK DAPAT DIJALANKAN!")
        print("Langkah-langkah:")
        print("1. Download dataset dari Kaggle")
        print("2. Jalankan: python prepare_data.py")
        print("3. Jalankan: python train.py")
        print("4. Jalankan: python app.py")
