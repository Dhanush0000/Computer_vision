from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy

import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
from PIL import Image
import numpy as np
import base64
import io

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///grocery_scanner.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

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
resnet_model_20 = models.resnet101(weights=None)
resnet_model_20.fc = nn.Linear(resnet_model_20.fc.in_features, len(class_names_20))
resnet_model_20.load_state_dict(torch.load('resnet101_trained.pth', map_location=device, weights_only=True))
resnet_model_20 = resnet_model_20.to(device)
resnet_model_20.eval()

# Load the 7-class ResNet model (PyTorch)
resnet_model_7 = models.resnet101(weights=None)
resnet_model_7.fc = nn.Linear(resnet_model_7.fc.in_features, len(class_names_7))
resnet_model_7.load_state_dict(torch.load('resnet101_trained_gro.pth', map_location=device, weights_only=True))
resnet_model_7 = resnet_model_7.to(device)
resnet_model_7.eval()

# Database models
class ScanResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    item_name = db.Column(db.String(100), nullable=False)
    freshness = db.Column(db.String(50), nullable=False)  # "fresh" or "rotten"
    category = db.Column(db.String(50), nullable=True)    # "fruit", "vegetable", etc.

class Category(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)

# Preprocessing for images (PyTorch model)
preprocess_torch = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Helper function to save results to database
def save_results_to_db(scan_type, predicted_class, confidence_score):
    new_scan = ScanResult(
        item_name=predicted_class,
        freshness="fresh" if "Fresh" in predicted_class else "rotten",
        category=scan_type
    )
    db.session.add(new_scan)
    db.session.commit()

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

# Helper function to process Base64 image
def process_base64_image(base64_image):
    img_data = base64.b64decode(base64_image.split(",")[1])
    image = Image.open(io.BytesIO(img_data))
    return np.array(image)

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
    predicted_class, confidence_score = predict_class_torch(image, resnet_model_20, class_names_20)

    # Save results to the database
    save_results_to_db("Fruits/Veggies", predicted_class, confidence_score)

    return jsonify({"message": f"Predicted Class: {predicted_class}, Confidence: {confidence_score:.2f}"})

# Route to handle scanning packaged products (7-class model, PyTorch)
@app.route('/scan_packaged_products', methods=['POST'])
def scan_packaged_products():
    try:
        data = request.json.get("imageData")
        image = process_base64_image(data)

        # Predict using the 7-class model (PyTorch)
        predicted_class, confidence_score = predict_class_torch(image, resnet_model_7, class_names_7)

        # Save results to the database
        save_results_to_db("Packaged Products", predicted_class, confidence_score)

        return jsonify({"message": f"Predicted Class: {predicted_class}, Confidence: {confidence_score:.2f}"})

    except Exception as e:
        print(f"Error in scan_packaged_products: {e}")
        return jsonify({"message": "Scan failed"}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
