import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# ----------------------------------------------------------------------------------------------------------------------
# 1)Hyper-parameters:
num_epochs = 20
batch_size = 1000
learning_rate = 0.001
# ----------------------------------------------------------------------------------------------------------------------
# 2)Dataset:
train_set = datasets.FashionMNIST(root= "./data", train= True, download= True,
                                         transform= transforms.Compose([transforms.RandomRotation(10), transforms.ToTensor()]))
test_set = datasets.FashionMNIST(root= "./data", train= False, download= True,
                                         transform= transforms.Compose([transforms.ToTensor()]))
# ----------------------------------------------------------------------------------------------------------------------
# 3)DataLoader
train_loader = DataLoader(dataset= train_set, batch_size= batch_size, shuffle= True)
test_loader = DataLoader(dataset= test_set, batch_size= batch_size, shuffle= False)
# ----------------------------------------------------------------------------------------------------------------------
# 4)Model:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
network1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
                         nn.ReLU(),
                         nn.MaxPool2d(kernel_size=2, stride=2),
                         nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5),
                         nn.ReLU(),
                         nn.MaxPool2d(kernel_size=2, stride=2),
                         nn.Flatten(start_dim=1),
                         nn.Linear(in_features=12 *4*4, out_features=120),
                         nn.ReLU(),
                         nn.Linear(in_features=120, out_features=60),
                         nn.ReLU(),
                         nn.Linear(in_features=60, out_features=10))
network2 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
                         nn.BatchNorm2d(6),
                         nn.ReLU(),
                         nn.MaxPool2d(kernel_size=2, stride=2),
                         nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5),
                         nn.BatchNorm2d(12),
                         nn.ReLU(),
                         nn.MaxPool2d(kernel_size=2, stride=2),
                         nn.Flatten(start_dim=1),
                         nn.Linear(in_features=12*4*4, out_features=120),
                         nn.BatchNorm1d(120),
                         nn.ReLU(),
                         nn.Dropout(0.3),
                         nn.Linear(in_features=120, out_features=60),
                         nn.ReLU(),
                         nn.Linear(in_features=60, out_features=10))
model1= network1.to(device)
model2 = network2.to(device)
# ----------------------------------------------------------------------------------------------------------------------
# 5)Loss & Optimizer:
criterion = nn.CrossEntropyLoss()
optimizer1 = torch.optim.Adam(params= model1.parameters(), lr= learning_rate)
optimizer2 = torch.optim.Adam(params= model2.parameters(), lr= learning_rate)
# ----------------------------------------------------------------------------------------------------------------------
# 6)Training Loop:
def train(model, optimizer):
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
print("Model 1:")
train(model1, optimizer1)
print("------------------------------------------------------------------------------")
print("Model 2:")
train(model2, optimizer2)
