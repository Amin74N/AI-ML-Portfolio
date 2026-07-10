import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
# 1)Dataset:
train_dataset = datasets.MNIST(root= "./data", train= True, transform= transforms.ToTensor(), download= True)
test_dataset = datasets.MNIST(root= "./data", train= False, transform= transforms.ToTensor(), download= True)
# ----------------------------------------------------------------------------------------------------------------------
# 2) DataLoader
train_loader = DataLoader(dataset= train_dataset, batch_size= 64, shuffle= True)
test_loader = DataLoader(dataset= test_dataset, batch_size= 64, shuffle= False)
# ----------------------------------------------------------------------------------------------------------------------
# 3)Model:
model = nn.Sequential(nn.Flatten(),
                      nn.Linear(784,128),
                      nn.ReLU(),
                      nn.Linear(128,64),
                      nn.ReLU(),
                      nn.Linear(64,10))
# ----------------------------------------------------------------------------------------------------------------------
# 4)Loss & Optimizer:
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr= 0.001)
# ----------------------------------------------------------------------------------------------------------------------
num_epochs = 5
for epoch in range(num_epochs):
# 5)Training Loop:
    model.train()
    epoch_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= len(train_loader)
# 6)Evaluation:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'epoch: {epoch + 1}/{num_epochs} | loss = {epoch_loss:.4f} | Accuracy = {accuracy:.2f}%')
# ----------------------------------------------------------------------------------------------------------------------
# 7)Prediction:
model.eval()
index = random.randint(0, len(test_dataset)-1)
with torch.no_grad():
    image,label = test_dataset[index]
    image = image.unsqueeze(0)
    output = model(image)
    _, prediction = torch.max(output,1)
plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Image {index}'s True Label: {label} & Prediction: {prediction.item()}")
plt.show()
