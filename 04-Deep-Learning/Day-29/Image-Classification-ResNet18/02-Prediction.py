import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
# 1)Loading Model:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights= None)
model.fc = nn.Linear(model.fc.in_features, 2)
checkpoint = torch.load("best_model.pth", map_location= device)
model.load_state_dict(checkpoint["model_state_dict"])
classes = checkpoint["classes"]
model = model.to(device)
model.eval()
# ----------------------------------------------------------------------------------------------------------------------
# 2)Data Preprocessing:
transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(
                                                                                 mean= [0.485, 0.456, 0.406],
                                                                                 std= [0.229, 0.224, 0.225])])
# ----------------------------------------------------------------------------------------------------------------------
# 3)Prediction Function:
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    with torch.no_grad():
        input_image = image.unsqueeze(0).to(device)
        output = model(input_image)
        probabilities = F.softmax(output,1)
        confidence, prediction = torch.max(probabilities, 1)
        confidence = confidence.item() * 100
        prediction = prediction.item()
    image = image.cpu()
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    image = image * std + mean
    image = image.clamp(0,1)
    plt.figure(figsize=(8,8))
    plt.imshow(image.permute(1,2,0))
    plt.title(f'Prediction: {classes[prediction]} ({confidence:.2f}%)')
    plt.axis("off")
    plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# 4) Random Image Prediction:
def predict_random_test_image(test_folder):
    classes_folder = os.listdir(test_folder)
    random_class = random.choice(classes_folder)
    image_folder = os.path.join(test_folder, random_class)
    random_image = random.choice(os.listdir(image_folder))
    image_path = os.path.join(image_folder, random_image)
    predict_image(image_path)
predict_random_test_image("data/Cats-vs-Dogs/test_set")
