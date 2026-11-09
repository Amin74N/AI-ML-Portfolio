import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
# 1)Hyper-parameters:
num_epochs = 20
batch_size = 32
learning_rate = 0.001
# ----------------------------------------------------------------------------------------------------------------------
# 2)Dataset:
classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
train_dataset = datasets.CIFAR10(root = './data', train = True, download = True,
                                 transform = transforms.Compose([transforms.RandomCrop(32, padding= 4),
                                                                 transforms.RandomHorizontalFlip(),
                                                                 transforms.ToTensor(),
                                                                 transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]))
test_dataset = datasets.CIFAR10(root = './data', train = False, download = True,
                                 transform = transforms.Compose([transforms.ToTensor(),
                                                                 transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]))
# ----------------------------------------------------------------------------------------------------------------------
# 3)Data-Loader:
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)
# ----------------------------------------------------------------------------------------------------------------------
# 4)Model:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels= 3, out_channels= 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels= 32, out_channels= 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size= 2, stride= 2)
        self.fc1 = nn.Linear(in_features= 64*8*8, out_features= 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(in_features= 512, out_features= 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(in_features= 256, out_features= 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(input= x, start_dim= 1)
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout2(x)
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
        epoch_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    epoch_loss /= train_total
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
# 8)Prediction & Augmentation Demo::
model.eval()
index = random.randint(0, len(test_dataset)-1)
crop = transforms.RandomCrop(32, padding= 4)
flip = transforms.RandomHorizontalFlip(p= 1.0)
with torch.no_grad():
    image,label = test_dataset[index]
    crop_image = crop(image * 0.5 + 0.5)
    flip_image = flip(image * 0.5 + 0.5)
    image = image.unsqueeze(0).to(device)
    output = model(image)
    _, prediction = torch.max(output,1)
image = (image * 0.5 + 0.5).squeeze().cpu()
plt.figure(figsize=(8,8))
plt.subplot(2, 2, 1)
plt.title(f"Image {index} | True Label: {classes[label]} | Prediction: {classes[prediction.item()]}")
plt.axis("off")
plt.subplot(2, 2, 2)
plt.imshow(image.permute(1,2,0))
plt.title("Real Image")
plt.axis("off")
plt.subplot(2, 2, 3)
plt.imshow(crop_image.permute(1,2,0))
plt.title("Crop Image")
plt.axis("off")
plt.subplot(2, 2, 4)
plt.imshow(flip_image.permute(1,2,0))
plt.title("Flip Image")
plt.axis("off")
plt.tight_layout()
plt.show()
