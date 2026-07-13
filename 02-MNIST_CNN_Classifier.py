import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
# 1)Hyper-parameters:
num_epochs = 5
batch_size = 64
learning_rate = 0.001
# ----------------------------------------------------------------------------------------------------------------------
# 2)Dataset:
train_dataset = datasets.MNIST(root= "./data", train= True, transform= transforms.ToTensor(), download= True)
test_dataset = datasets.MNIST(root= "./data", train= False, transform= transforms.ToTensor(), download= True)
# ----------------------------------------------------------------------------------------------------------------------
# 3) DataLoader
train_loader = DataLoader(dataset= train_dataset, batch_size= batch_size, shuffle= True)
test_loader = DataLoader(dataset= test_dataset, batch_size= batch_size, shuffle= False)
# ----------------------------------------------------------------------------------------------------------------------
# 4)Model:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels= 1, out_channels= 16, kernel_size= 3)
        self.conv2 = nn.Conv2d(in_channels= 16, out_channels= 32, kernel_size= 3)
        self.pool = nn.MaxPool2d(kernel_size= 2, stride= 2)
        self.fc1 = nn.Linear(in_features= 32*5*5,out_features= 128)
        self.fc2 = nn.Linear(in_features= 128, out_features= 64)
        self.fc3 = nn.Linear(in_features= 64, out_features= 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(input= x, start_dim= 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
model = CNN().to(device)
# ----------------------------------------------------------------------------------------------------------------------
# 5)Loss & Optimizer:
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params= model.parameters(), lr= learning_rate)
# ----------------------------------------------------------------------------------------------------------------------
# 6)Training Loop:
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    train_correct = 0
    train_total = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    epoch_loss /= len(train_loader)
    train_accuracy = 100 * train_correct / train_total
# 7)Evaluation:
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    test_accuracy = 100 * test_correct / test_total
    print(f'epoch: {epoch + 1}/{num_epochs} | loss = {epoch_loss:.4f} | Train Accuracy = {train_accuracy:.2f}% | Test Accuracy = {test_accuracy:.2f}%')
# ----------------------------------------------------------------------------------------------------------------------
# 8)Prediction:
model.eval()
index = random.randint(0, len(test_dataset)-1)
with torch.no_grad():
    image,label = test_dataset[index]
    image = image.unsqueeze(0).to(device)
    output = model(image)
    _, prediction = torch.max(output,1)
plt.imshow(image.cpu.squeeze(), cmap="gray")
plt.title(f"Image {index}'s True Label: {label} & Prediction: {prediction.item()}")
plt.show()
