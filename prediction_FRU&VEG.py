import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
from PIL import Image
import numpy as np

# Load the saved ResNet101 model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_classes = 20  # Set the number of classes based on your dataset

# Define the model and load the trained weights
model_ft = models.resnet101(pretrained=False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, num_classes)  # Adjust for your number of classes
model_ft.load_state_dict(torch.load('resnet101_trained.pth'))  # Load the trained model
model_ft = model_ft.to(device)
model_ft.eval()  # Set the model to evaluation mode

# Define the image preprocessing transforms (same as during training)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the class names (adjust according to your dataset)
class_names = ['FreshApple', 'FreshBanana', 'FreshBellpepper', 'FreshCarrot', 'FreshCucumber', 'FreshMango', 'FreshOrange', 'FreshPotato', 'FreshStrawberry', 'FreshTomato' ,'RottenApple','RottenBanana','RottenBellpepper','RottenCarrot','RottenCucumber','RottenMango','RottenOrange','RottenPotato','RottenStrawberry','RottenTomato']


# Function to preprocess a frame and make predictions
def predict_class(frame, model):
    # Convert the frame from BGR (OpenCV format) to RGB (PIL format)
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Preprocess the image (resize, normalize, etc.)
    image = preprocess(image).unsqueeze(0).to(device)  # Add batch dimension and send to GPU

    # Forward pass through the model to get predictions
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)  # Get the class with the highest score
        predicted_class = class_names[preds[0].item()]

    return predicted_class


# Open the webcam or video feed
cap = cv2.VideoCapture(0)  # Use '0' for the default webcam

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Loop to read frames and make predictions
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Predict the class of the current frame
    predicted_class = predict_class(frame, model_ft)

    # Display the predicted class on the frame
    cv2.putText(frame, f"Predicted: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                cv2.LINE_AA)

    # Display the frame with the prediction
    cv2.imshow('Live Feed - Press "q" to Quit', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()

