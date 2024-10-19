from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
from PIL import Image
import os
import numpy as np
import base64
import io
from tensorflow.keras.models import load_model
from datetime import datetime

app = Flask(__name__)

# Device configuration for PyTorch (for ResNet model)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Class names for both models
class_names_20 = ['FreshApple', 'FreshBanana', 'FreshBellpepper', 'FreshCarrot', 'FreshCucumber', 'FreshMango',
                  'FreshOrange', 'FreshPotato', 'FreshStrawberry', 'FreshTomato', 'RottenApple', 'RottenBanana',
                  'RottenBellpepper', 'RottenCarrot', 'RottenCucumber', 'RottenMango', 'RottenOrange', 'RottenPotato',
                  'RottenStrawberry', 'RottenTomato']
class_names_7 = ['Dairy and Eggs', 'HouseHold Care', 'Kitchen_products', 'Packaged Foods', 'Personal and Baby Care',
                 'Snack and Beverages', 'Staples']

# Load the 20-class ResNet model (PyTorch)
resnet_model = models.resnet101(weights=None)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, len(class_names_20))
resnet_model.load_state_dict(torch.load('resnet101_trained.pth', map_location=device))
resnet_model = resnet_model.to(device)
resnet_model.eval()

# Load the 7-class EfficientNet model (Keras/TensorFlow)
efficientnet_model = load_model('efficientnet_finetuned_model_grocery2.keras')

# Preprocessing for images (PyTorch model)
preprocess_torch = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Preprocessing for images (Keras model)
preprocess_keras = lambda img: np.expand_dims(cv2.resize(img, (224, 224)) / 255.0, axis=0)

# Prediction function for PyTorch (ResNet, 20-class model)
def predict_class_torch(image, model, class_names):
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = preprocess_torch(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        softmax_scores = torch.nn.functional.softmax(outputs, dim=1)
        confidence, preds = torch.max(softmax_scores, 1)
        predicted_class = class_names[preds[0].item()]
        confidence_score = confidence[0].item()
    return predicted_class, confidence_score

# Prediction function for Keras (EfficientNet, 7-class model)
def predict_class_keras(image, model, class_names):
    image = preprocess_keras(image)
    predictions = model.predict(image)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_names[predicted_class_index]
    confidence_score = np.max(predictions)
    return predicted_class, confidence_score

# Helper function to save results to a file
def save_results_to_file(predicted_class, confidence_score):
    with open('results.txt', 'a') as f:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"{timestamp} - Class: {predicted_class}, Confidence: {confidence_score:.2f}\n")

# Route to the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle scanning fruits and vegetables (20-class model, PyTorch)
@app.route('/scan_fruits_veggies', methods=['POST'])
def scan_fruits_veggies():
    data = request.json.get("imageData")
    image = process_base64_image(data)

    # Predict using the 20-class model (PyTorch)
    predicted_class, confidence_score = predict_class_torch(image, resnet_model, class_names_20)
    
    # Save results to results.txt
    save_results_to_file(predicted_class, confidence_score)

    return jsonify({"message": f"Predicted Class: {predicted_class}, Confidence: {confidence_score:.2f}"})

# Route to handle scanning packaged products (7-class model, Keras)
@app.route('/scan_packaged_products', methods=['POST'])
def scan_packaged_products():
    data = request.json.get("imageData")
    image = process_base64_image(data)

    # Predict using the 7-class model (Keras)
    predicted_class, confidence_score = predict_class_keras(image, efficientnet_model, class_names_7)

    # Save results to results.txt
    save_results_to_file(predicted_class, confidence_score)

    return jsonify({"message": f"Predicted Class: {predicted_class}, Confidence: {confidence_score:.2f}"})

# Function to process the base64-encoded image data sent from the camera feed
def process_base64_image(base64_image):
    img_data = base64.b64decode(base64_image.split(",")[1])
    image = Image.open(io.BytesIO(img_data))
    return np.array(image)

if __name__ == '__main__':
    app.run(debug=True)
