import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
# 1)Hyper-parameters:
num_epochs = 20
batch_size = 32
learning_rate = 0.001
# ----------------------------------------------------------------------------------------------------------------------
# 2)Dataset:
train_transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),transforms.Normalize(mean= [0.485, 0.456, 0.406],
                                                                                 std= [0.229, 0.224, 0.225])])
test_transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(
                                                                                 mean= [0.485, 0.456, 0.406],
                                                                                 std= [0.229, 0.224, 0.225])])
train_dataset = datasets.ImageFolder(root= "data/Cats-vs-Dogs/training_set", transform= train_transform)
test_dataset = datasets.ImageFolder(root= "data/Cats-vs-Dogs/test_set", transform= test_transform)
classes = train_dataset.classes
# ----------------------------------------------------------------------------------------------------------------------
# 3)Data-Loader:
train_loader = DataLoader(dataset= train_dataset, batch_size= batch_size, shuffle= True)
test_loader = DataLoader(dataset= test_dataset, batch_size= batch_size, shuffle= False)
# ----------------------------------------------------------------------------------------------------------------------
# 4)ResNet18 Feature Extraction Transfer Learning Model:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights= "DEFAULT")
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# ----------------------------------------------------------------------------------------------------------------------
# 5)Loss, Optimizer & Scheduler:
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr= learning_rate)
scheduler = StepLR(optimizer, step_size= 5, gamma= 0.1)
# ----------------------------------------------------------------------------------------------------------------------
# 6)Training Loop:
start_time = time.time()
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
best_accuracy = 0.0
best_epoch = 0
model.fc.train()
for epoch in range(num_epochs):
    model.eval()
    model.fc.train()
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
    train_losses.append(epoch_loss)
    train_accuracies.append(train_accuracy)
# ----------------------------------------------------------------------------------------------------------------------
# 7)Evaluation:
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    test_loss /= test_total
    test_accuracy = 100 * test_correct / test_total
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_epoch = epoch + 1
        torch.save({"model_state_dict": model.state_dict(), "classes": classes}, "best_model.pth")
    print(f'epoch: {epoch + 1}/{num_epochs} | LR = {optimizer.param_groups[0]["lr"]:.6f} | Train Loss = {epoch_loss:.4f} | Train Accuracy = {train_accuracy:.2f}% | Test Loss = {test_loss:.4f} | Test Accuracy = {test_accuracy:.2f}%')
    scheduler.step()
training_time = time.time() - start_time
# ----------------------------------------------------------------------------------------------------------------------
# 8)Loading Best Model & Final Evaluation:
checkpoint = torch.load("best_model.pth", map_location= device)
model.load_state_dict(checkpoint["model_state_dict"])
classes = checkpoint["classes"]
model.eval()
all_labels = []
all_predictions = []
final_correct = 0
final_total = 0
final_loss = 0.0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        loss = criterion(outputs, labels)
        final_loss += loss.item() * images.size(0)
        final_total += labels.size(0)
        final_correct += (predicted == labels).sum().item()
final_accuracy = 100 * final_correct / final_total
final_loss /= final_total
# ----------------------------------------------------------------------------------------------------------------------
# 9)Metrics & Confusion Matrix:
precision = precision_score(all_labels, all_predictions, zero_division= 0)
recall = recall_score(all_labels, all_predictions, zero_division= 0)
f1 = f1_score(all_labels, all_predictions, zero_division= 0)
cm = confusion_matrix(all_labels, all_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix= cm, display_labels= classes)
fig, ax = plt.subplots(figsize= (7, 7))
disp.plot(ax= ax, cmap= "Blues", xticks_rotation= 45)
ax.set_title("Confusion Matrix - Best Model")
plt.savefig("Confusion_Matrix.png")
plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# 10)Model Reports:
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f'Accuracy: {final_accuracy:.2f}%')
print(f'Loss: {final_loss:.4f}')
print(f'Training Time: {training_time:.2f} seconds')
print(f'Trainable Parameters: {trainable_params:,}')
epochs = range(1, len(train_losses) + 1)
plt.figure(figsize=(8,5))
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, test_losses, label="Test Loss")
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("Loss_Curve.png")
plt.show()
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_accuracies, label="Train Accuracy")
plt.plot(epochs, test_accuracies, label="Test Accuracy")
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True)
plt.savefig("Accuracy_Curve.png")
plt.show()
