import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
# ----------------------------------------------------------------------------------------------------------------------
# 1)Hyper-parameters:
num_epochs = 15
batch_size = 32
learning_rate = 0.01
# ----------------------------------------------------------------------------------------------------------------------
# 2)Dataset:
classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
### Transforming PILImages of range [0,1] to Tensors of normalized range [-1,1]:
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
train_dataset = torchvision.datasets.CIFAR10(root = './data', train = True, transform = transform, download = True)
test_dataset = torchvision.datasets.CIFAR10(root = './data', train = False, transform = transform, download = True)
# ----------------------------------------------------------------------------------------------------------------------
# 3)Data-Loader:
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)
# ----------------------------------------------------------------------------------------------------------------------
# 4)Model:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
model = ConvNet().to(device)
# ----------------------------------------------------------------------------------------------------------------------
# 5)Loss & Optimizer:
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# ----------------------------------------------------------------------------------------------------------------------
# 6)Training Loop:
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        ### origin shape: [32,3,32,32]
        ### input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)
        ### Forward pass:
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        ### Backward and optimize:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
# ----------------------------------------------------------------------------------------------------------------------
# 7)Evaluation:
    model.eval()
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(10)]
        n_class_samples = [0 for i in range(10)]
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            ### Max returns (value, index):
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()
            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = predicted[i].item()
                if (label == pred):
                    n_class_correct[label] += 1
                n_class_samples[label] += 1
        acc = 100 * n_correct / n_samples
        print(f'epoch: {epoch + 1}/{num_epochs} | loss = {running_loss / len(train_loader):.4f} | Accuracy: {acc:.2f}%')
print("-----------------------------------------------------")
for i in range(10):
    acc = 100.0 * n_class_correct[i] / n_class_samples[i]
    print(f'Accuracy of {classes[i]}: {acc:.2f}%')
