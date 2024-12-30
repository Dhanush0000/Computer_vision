import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
from PIL import Image
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_classes = 20

model_ft = models.resnet101(pretrained=False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, num_classes)
model_ft.load_state_dict(torch.load('resnet101_trained.pth'))
model_ft = model_ft.to(device)
model_ft.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class_names = ['FreshApple', 'FreshBanana', 'FreshBellpepper', 'FreshCarrot', 'FreshCucumber', 'FreshMango',
               'FreshOrange', 'FreshPotato', 'FreshStrawberry', 'FreshTomato', 'RottenApple', 'RottenBanana',
               'RottenBellpepper', 'RottenCarrot', 'RottenCucumber', 'RottenMango', 'RottenOrange', 'RottenPotato',
               'RottenStrawberry', 'RottenTomato']

class_count = {class_name: 0 for class_name in class_names}
previous_class = None  # Store the previously predicted class

# Store results for the txt file
results = {}


def predict_class(frame, model):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        softmax_scores = torch.nn.functional.softmax(outputs, dim=1)
        confidence, preds = torch.max(softmax_scores, 1)
        predicted_class = class_names[preds[0].item()]
        confidence_score = confidence[0].item()
    return predicted_class, confidence_score


cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    predicted_class, confidence_score = predict_class(frame, model_ft)

    # Only increment count if the predicted class changes
    if predicted_class != previous_class:
        class_count[predicted_class] += 1
        previous_class = predicted_class

    # Store the results
    results[predicted_class] = {"confidence": confidence_score, "count": class_count[predicted_class]}

    cv2.putText(frame, f"Predicted: {predicted_class} ({confidence_score * 100:.2f}%)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Live Feed - Press "q" to Quit', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Write the results to the results.txt file in table format
with open(r'C:\Users\traks\PycharmProjects\pythonProject\results.txt', 'w') as f:
    f.write("Class\tCount\tConfidence Score\n")
    for class_name, data in results.items():
        f.write(f"{class_name}\t{data['count']}\t{data['confidence'] * 100:.2f}%\n")

cap.release()

