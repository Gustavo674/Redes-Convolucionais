from flask import request, jsonify, render_template, redirect, url_for
from app import app
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import joblib

model_cnn = load_model('model/pesos.h5')
model_linear = joblib.load('model/model_linear.pkl')

@app.route('/')
def index():
    return redirect(url_for('upload'))

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = Image.open(file).convert('L').resize((28, 28))
    img = np.array(img) / 255.0
    img = img.reshape(1, 28, 28, 1)
    
    # Predição com CNN
    prediction_cnn = model_cnn.predict(img)
    digit_cnn = np.argmax(prediction_cnn)
    
    # Predição com modelo linear
    img_linear = img.reshape(1, -1)
    digit_linear = model_linear.predict(img_linear)[0]
    
    return jsonify({'digit_cnn': int(digit_cnn), 'digit_linear': int(digit_linear)})

@app.route('/upload')
def upload():
    return render_template('upload.html')
