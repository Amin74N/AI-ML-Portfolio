import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
# 1)Hyper-parameters:
num_epochs = 20
batch_size = 32
learning_rate = 0.01
# ----------------------------------------------------------------------------------------------------------------------
# 2)Dataset:
classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
train_dataset = datasets.CIFAR10(root= './data', train= True, download= True,
                                 transform = transforms.Compose([transforms.RandomCrop(32, padding= 4),
                                                                 transforms.RandomHorizontalFlip(),
                                                                 transforms.Resize((224, 224)),
                                                                 transforms.ToTensor(),
                                                                 transforms.Normalize(mean= [0.485, 0.456, 0.406],
                                                                                      std= [0.229, 0.224, 0.225])]))
test_dataset = datasets.CIFAR10(root= './data', train= False, download= True,
                                 transform = transforms.Compose([transforms.Resize((224, 224)),
                                                                 transforms.ToTensor(),
                                                                 transforms.Normalize(mean= [0.485, 0.456, 0.406],
                                                                                      std= [0.229, 0.224, 0.225])]))
# ----------------------------------------------------------------------------------------------------------------------
# 3)Data-Loader:
train_loader = DataLoader(dataset= train_dataset, batch_size= batch_size, shuffle= True)
test_loader = DataLoader(dataset= test_dataset, batch_size= batch_size, shuffle= False)
# ----------------------------------------------------------------------------------------------------------------------
# 4)Model:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights= "DEFAULT")
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 10)
for param in model.layer4.parameters():
    param.requires_grad = True
model = model.to(device)
# ----------------------------------------------------------------------------------------------------------------------
# 5)Loss, Optimizer & Scheduler:
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr= learning_rate)
scheduler = StepLR(optimizer, step_size= 5, gamma= 0.1)
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
    print(f'epoch: {epoch + 1}/{num_epochs} | LR = {optimizer.param_groups[0]["lr"]:.5f} | loss = {epoch_loss:.4f} | Train Accuracy = {train_accuracy:.2f}% | Test Accuracy = {test_accuracy:.2f}%')
    scheduler.step()
# ----------------------------------------------------------------------------------------------------------------------
# 8)Prediction & Augmentation Demo:
model.eval()
index = random.randint(0, len(test_dataset)-1)
crop = transforms.RandomCrop(32, padding= 4)
flip = transforms.RandomHorizontalFlip(p= 1.0)
mean = torch.tensor([0.485 ,0.456 ,0.406]).view(3, 1, 1)
std  = torch.tensor([0.229 ,0.224 ,0.225]).view(3, 1, 1)
with torch.no_grad():
    image,label = test_dataset[index]
    visual_crop = crop(image * std + mean)
    visual_flip = flip(image * std + mean)
    image = image.unsqueeze(0).to(device)
    output = model(image)
    _, prediction = torch.max(output,1)
image = image.squeeze().cpu()
image = image * std + mean
plt.figure(figsize=(8, 8))
plt.subplot(2, 2, 1)
plt.title(f"Image {index} | True Label: {classes[label]} | Prediction: {classes[prediction.item()]}")
plt.axis("off")
plt.subplot(2, 2, 2)
plt.imshow(image.permute(1, 2, 0))
plt.title("Real Image")
plt.axis("off")
plt.subplot(2, 2, 3)
plt.imshow(visual_crop.permute(1, 2, 0))
plt.title("Crop Image")
plt.axis("off")
plt.subplot(2, 2, 4)
plt.imshow(visual_flip.permute(1, 2, 0))
plt.title("Flip Image")
plt.axis("off")
plt.tight_layout()
plt.show()
