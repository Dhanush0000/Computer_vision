from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy

import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import os
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

# Helper function to load and modify the model
def load_improved_model(num_classes, model_path):
    model_ft = models.resnet101(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)  # Adjust for your number of classes

    # Load the trained model weights
    state_dict = torch.load(model_path, map_location=device)
    model_ft.load_state_dict(state_dict)

    return model_ft.to(device).eval()

# Initialize both models with appropriate class counts and weights
resnet_model_20 = load_improved_model(len(class_names_20), 'resnet101_trained.pth')
resnet_model_7 = load_improved_model(len(class_names_7), 'resnet101_trained_gro.pth')

# Database models
class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    barcode = db.Column(db.String(100), unique=True, nullable=False)

    def __repr__(self):
        return f'<Product {self.name}>'

class ScanResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    item_name = db.Column(db.String(100), nullable=False)
    freshness = db.Column(db.String(50), nullable=False)  # "fresh" or "rotten"
    category = db.Column(db.String(50), nullable=True)    # "fruit", "vegetable", etc.
    file_path = db.Column(db.String(200), nullable=False) # Path to the results file

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

import sqlite3

def save_results_to_file_and_db(scan_type, predicted_class, confidence_score):
    try:
        # Connect to the SQLite database
        db_path = 'result1.db'
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()

        # Create the table if it doesn't already exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scan_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_type TEXT,
                predicted_class TEXT,
                confidence_score REAL
            )
        ''')

        # Insert the results into the table
        cursor.execute('''
            INSERT INTO scan_results (scan_type, predicted_class, confidence_score)
            VALUES (?, ?, ?)
        ''', (scan_type, predicted_class, confidence_score))

        # Commit the transaction and close the connection
        connection.commit()
        connection.close()

        print("Results saved to database successfully.")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"General error: {e}")

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

    # Save results to file and database
    save_results_to_file_and_db("Fruits/Veggies", predicted_class, confidence_score)

    return jsonify({"message": f"Predicted Class: {predicted_class}, Confidence: {confidence_score:.2f}"})

# Route to handle scanning packaged products (7-class model, PyTorch)
@app.route('/scan_packaged_products', methods=['POST'])
def scan_packaged_products():
    try:
        data = request.json.get("imageData")
        image = process_base64_image(data)

        # Predict using the 7-class model (PyTorch)
        predicted_class, confidence_score = predict_class_torch(image, resnet_model_7, class_names_7)

        # Save results to file and database
        save_results_to_file_and_db("Packaged Products", predicted_class, confidence_score)

        return jsonify({"message": f"Predicted Class: {predicted_class}, Confidence: {confidence_score:.2f}"})

    except Exception as e:
        print(f"Error in scan_packaged_products: {e}")
        return jsonify({"message": "Scan failed"}), 500

# Route to handle product scanning (POST request to add a product to the database)
@app.route('/scan', methods=['POST'])
def scan_product():
    data = request.json  # Assuming JSON data is sent with product details
    try:
        new_product = Product(name=data['name'], barcode=data['barcode'])
        db.session.add(new_product)
        db.session.commit()
        return {'message': 'Product saved successfully!'}, 201
    except Exception as e:
        db.session.rollback()
        return {'message': f'Error saving product: {str(e)}'}, 500

#for database download still testing
@app.route('/download_db')
def download_db():
    db_path = 'result1.db'  # Ensure this is the correct path
    if os.path.exists(db_path):
        return send_file(db_path, as_attachment=True)  # Enable download
    else:
        return "Database file not found", 404

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
