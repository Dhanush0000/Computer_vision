import os  # <-- We need os for file handling
import cv2
import pytesseract
import re
from dateutil import parser as dateparser
from datetime import datetime
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from collections import Counter

# Set the path to the Tesseract executable (on Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Device for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load EfficientNet and modify the final layer to match the number of classes
efficientnet = models.efficientnet_b0(weights='EfficientNet_B0_Weights.IMAGENET1K_V1')

# Number of classes (7 for your case)
num_classes = 7
efficientnet.classifier[1] = nn.Linear(efficientnet.classifier[1].in_features, num_classes)

# Move the model to the appropriate device (GPU or CPU)
efficientnet = efficientnet.to(device)
efficientnet.eval()  # Set model to evaluation mode

# List of class names
class_names = ['Dairy and Eggs', 'HouseHold Care', 'Kitchen_products', 'Packaged Foods', 'Personal and Baby Care',
               'Snack and Beverages', 'Staples']

# Define the image transformations (same as validation)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Function to predict the class of a frame
def predict_frame(frame):
    # Convert the frame to a PIL image
    image = Image.fromarray(frame)

    # Apply transformations
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Move image to the same device as the model
    image = image.to(device)

    # Disable gradient calculations (since we're doing inference)
    with torch.no_grad():
        # Forward pass through the model
        outputs = efficientnet(image)

        # Get the predicted class (the index with the highest output score)
        _, predicted_class = torch.max(outputs, 1)

    # Map the predicted index to the actual class name
    predicted_class_name = class_names[predicted_class.item()]
    print(f"[INFO] Predicted Class: {predicted_class_name}")  # Debugging print
    return predicted_class_name


# OCR related functions
def tesseract_ocr_image(image):
    custom_config = r'--oem 3 --psm 6'  # Experimenting with a more powerful OCR mode
    return pytesseract.image_to_string(image, config=custom_config, lang='eng')  # Force English language


def clean_text(text):
    cleaned_text = re.sub(r'[^a-zA-Z0-9\/\-\s]', '', text)
    cleaned_text = cleaned_text.lower()
    return cleaned_text


# Improve preprocessing by adjusting filters and adding contrast enhancement
def preprocess_image(image, resize_dim=(640, 480)):
    resized_image = cv2.resize(image, resize_dim)

    # Convert to grayscale
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding for better OCR results
    thresh_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)

    # Sharpen the image to make the text clearer for OCR
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    sharpened_image = cv2.filter2D(thresh_image, -1, kernel)

    # Save the preprocessed image for debugging
    cv2.imwrite("preprocessed_image.png", sharpened_image)  # Check how it looks

    return sharpened_image


def extract_dates(text):
    manufacture_keywords = ['manufacture', 'mfg', 'packed on', 'mfd', 'prod date', 'prod', 'packed']
    expiry_keywords = ['expiry', 'exp', 'best before', 'use by', 'valid until', 'exp date', 'valid']

    date_patterns = [
        r'(\d{2}[\/\-]\d{2}[\/\-]\d{4})',
        r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})',
        r'(\d{4}[\/\-]\d{2}[\/\-]\d{2})',
        r'(\d{2}[\/\-]\d{4})',
        r'(\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4})',
        r'(\d{2}[ ](?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[ ]\d{4})',
        r'(\d{1,2}[\/\-]\d{4})'
    ]

    dates = {'manufacture': None, 'expiry': None}
    lines = text.split("\n")

    for line in lines:
        line_lower = line.lower()
        manufacture_found = any(keyword in line_lower for keyword in manufacture_keywords)
        expiry_found = any(keyword in line_lower for keyword in expiry_keywords)

        for pattern in date_patterns:
            matches = re.findall(pattern, line_lower)
            for match in matches:
                try:
                    date_obj = dateparser.parse(match, dayfirst=True)
                    if manufacture_found:
                        dates['manufacture'] = date_obj
                    elif expiry_found:
                        dates['expiry'] = date_obj
                except ValueError:
                    continue

    if dates['manufacture'] or dates['expiry']:
        return dates
    return None


def calculate_days_left(expiry_date):
    if expiry_date:
        current_date = datetime.now()
        days_left = (expiry_date - current_date).days
        return days_left if days_left > 0 else "Expired"
    return 'N/A'


# Write the predicted class, extracted text, and other information to the .txt file
def update_product_status_in_file(products_with_dates, predicted_class, brand_text, file_path):
    products = load_existing_products(file_path)

    # Even if no dates are found, still update the file
    manufacture_date = products_with_dates[0][1].get('manufacture') if products_with_dates else None
    expiry_date = products_with_dates[0][1].get('expiry') if products_with_dates else None
    days_left = calculate_days_left(expiry_date) if expiry_date else 'N/A'

    product_entry = {
        'product': predicted_class,  # Use the predicted class as the product category
        'manufacture_date': manufacture_date.strftime('%Y-%m-%d') if isinstance(manufacture_date, datetime) else 'N/A',
        'expiry_date': expiry_date.strftime('%Y-%m-%d') if isinstance(expiry_date, datetime) else 'N/A',
        'brand_or_text': brand_text,
        'count': 1,
        'days_left': days_left
    }

    if product_entry['product'] in products:
        products[product_entry['product']]['count'] += 1
    else:
        products[product_entry['product']] = product_entry

    print(f"[INFO] Writing data to {file_path}")  # Debugging print
    write_products_to_file(products, file_path)


def load_existing_products(file_path):
    products = {}
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()

            for line in lines[1:]:
                # Skip malformed lines that don't have enough columns
                columns = line.strip().split('|')
                if len(columns) < 5:
                    print(f"[WARNING] Skipping malformed line: {line.strip()}")
                    continue

                product = columns[0].strip()
                try:
                    count = int(columns[4].strip())
                except ValueError:
                    print(f"[ERROR] Invalid count value in line: {line.strip()}")
                    continue

                products[product] = {'count': count}

    return products


def write_products_to_file(products, file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_path, 'w') as f:
        f.write(
            f"{'Predicted Class':<30} | {'Manufacture Date':<15} | {'Expiry Date':<15} | {'Brand/Text':<20} | {'Count':<5} | {'Days Left':<10}\n")
        f.write('-' * 110 + '\n')
        for product, details in products.items():
            f.write(
                f"{product:<30} | {details['manufacture_date']:<15} | {details['expiry_date']:<15} | {details['brand_or_text']:<20} | {details['count']:<5} | {details['days_left']:<10}\n")


# Main script to process the frame, predict the class, extract dates, and update the text file
output_file_path = r"C:\Users\traks\PycharmProjects\Fruit_&_Vegitables\products.txt"  # Change to your desired output path
cap = cv2.VideoCapture(0)  # Webcam source

if not cap.isOpened():
    print("[ERROR] Could not open video capture")
else:
    print("[INFO] Video capture started")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to capture frame")
        break

    # Display the frame
    cv2.imshow('Frame', frame)

    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF

    # If the 'c' key is pressed, predict the class and run OCR
    if key == ord('c'):
        predicted_class = predict_frame(frame)

        # Process frame for OCR detection
        preprocessed_frame = preprocess_image(frame)
        extracted_text = tesseract_ocr_image(preprocessed_frame)
        cleaned_text = clean_text(extracted_text)

        print(f"[INFO] Extracted Text: {cleaned_text}")  # Debugging print

        products_with_dates = []
        dates_info = extract_dates(cleaned_text)
        if dates_info:
            products_with_dates.append((cleaned_text, dates_info))
            print(f"[INFO] Detected Dates: {dates_info}")  # Debugging print

        # Update the text file with the product info, even if no dates are found
        update_product_status_in_file(products_with_dates, predicted_class, cleaned_text, output_file_path)

    # If the 'q' key is pressed, exit the loop
    elif key == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
print("[INFO] Video capture stopped")
