import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import random
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
train_dataset = datasets.ImageFolder(root="Cats-vs-Dogs/training_set", transform= train_transform)
test_dataset = datasets.ImageFolder(root= "Cats-vs-Dogs/test_set", transform= test_transform)
classes = train_dataset.classes
# ----------------------------------------------------------------------------------------------------------------------
# 3)Data-Loader:
train_loader = DataLoader(dataset= train_dataset, batch_size= batch_size, shuffle= True)
test_loader = DataLoader(dataset= test_dataset, batch_size= batch_size, shuffle= False)
# ----------------------------------------------------------------------------------------------------------------------
# 4)Model:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FT_model = models.resnet18(weights= "DEFAULT")
for param in FT_model.parameters():
    param.requires_grad = False
FT_model.fc = nn.Linear(FT_model.fc.in_features, 2)
for param in FT_model.layer4.parameters():
    param.requires_grad = True
FT_model = FT_model.to(device)
FT_trainable_params = sum(p.numel() for p in FT_model.parameters() if p.requires_grad)
FE_model = models.resnet18(weights= "DEFAULT")
for param in FE_model.parameters():
    param.requires_grad = False
FE_model.fc = nn.Linear(FE_model.fc.in_features, 2)
FE_model = FE_model.to(device)
FE_trainable_params = sum(p.numel() for p in FE_model.parameters() if p.requires_grad)
# ----------------------------------------------------------------------------------------------------------------------
# 5)Loss, Optimizer & Scheduler:
criterion = nn.CrossEntropyLoss()
FT_optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad, FT_model.parameters()), lr= learning_rate)
FE_optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad, FE_model.parameters()), lr= learning_rate)
FT_scheduler = StepLR(FT_optimizer, step_size= 5, gamma= 0.1)
FE_scheduler = StepLR(FE_optimizer, step_size= 5, gamma= 0.1)
# ----------------------------------------------------------------------------------------------------------------------
# 6)Training Loop:
def runner (model, optimizer, scheduler):
    start_time = time.time()
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
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
        print(f'epoch: {epoch + 1}/{num_epochs} | LR = {optimizer.param_groups[0]["lr"]:.5f} | Train Loss = {epoch_loss:.4f} | Train Accuracy = {train_accuracy:.2f}% | Test Loss = {test_loss:.4f} | Test Accuracy = {test_accuracy:.2f}%')
        scheduler.step()
    training_time = time.time() - start_time
# ----------------------------------------------------------------------------------------------------------------------
# 8)Confusion Matrix:
    cm = confusion_matrix(all_labels, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix= cm, display_labels= classes)
    fig, ax = plt.subplots(figsize= (7, 7))
    disp.plot(ax= ax, cmap= "Blues", xticks_rotation= 45)
    ax.set_title("Confusion Matrix")
    plt.show()
    # ----------------------------------------------------------------------------------------------------------------------
# 9)Prediction:
    model.eval()
    index = random.randint(0, len(test_dataset)-1)
    mean = torch.tensor([0.485 ,0.456 ,0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229 ,0.224 ,0.225]).view(3, 1, 1)
    with torch.no_grad():
        image,label = test_dataset[index]
        image = image.unsqueeze(0).to(device)
        output = model(image)
        _, prediction = torch.max(output,1)
    image = image.squeeze().cpu()
    image = image * std + mean
    image = image.clamp(0, 1)
    plt.figure(figsize=(8, 8))
    plt.imshow(image.permute(1, 2, 0))
    plt.title(f"Image {index} | True Label: {classes[label]} | Prediction: {classes[prediction.item()]}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    return (test_accuracy, epoch_loss, training_time, train_losses, test_losses, train_accuracies, test_accuracies)
# ----------------------------------------------------------------------------------------------------------------------
# 10)Run:
print ("Feature Extraction Model:")
(FE_acc, FE_loss, FE_time, FE_train_losses, FE_test_losses, FE_train_accuracies, FE_test_accuracies) = runner(FE_model, FE_optimizer, FE_scheduler)
print(f'Accuracy: {FE_acc:.2f}%')
print(f'Loss: {FE_loss:.4f}')
print(f'Training Time: {FE_time:.2f} seconds')
print(f'Trainable Parameters: {FE_trainable_params:,}')
print("--------------------------------------------------------------------------------")
print ("Fine-Tuning Model:")
(FT_acc, FT_loss, FT_time, FT_train_losses, FT_test_losses, FT_train_accuracies, FT_test_accuracies) = runner(FT_model, FT_optimizer, FT_scheduler)
print(f'Accuracy: {FT_acc:.2f}%')
print(f'Loss: {FT_loss:.4f}')
print(f'Training Time: {FT_time:.2f} seconds')
print(f'Trainable Parameters: {FT_trainable_params:,}')
# ----------------------------------------------------------------------------------------------------------------------
# 11)Drawing Training Curves:
def draw_curves(train_losses, test_losses, train_accuracies, test_accuracies, title):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, test_losses, label="Test Loss")
    plt.title(f"{title} - Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, test_accuracies, label="Test Accuracy")
    plt.title(f"{title} - Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.show()
draw_curves(FE_train_losses, FE_test_losses, FE_train_accuracies, FE_test_accuracies, "Feature Extraction")
draw_curves( FT_train_losses, FT_test_losses, FT_train_accuracies, FT_test_accuracies, "Fine-Tuning")
