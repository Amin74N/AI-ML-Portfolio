from pathlib import Path
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
# ----------------------------------------------------------------------------------------------------------------------
# 1)Hyper-Parameters:
DATASET_PATH = Path("data/UrbanSound8K")
ANNOTATION_FILE = DATASET_PATH / "UrbanSound8K.csv"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------------------------------------------------------------------------------------------------
# 2)Dataset:
class UrbanSoundDataset(Dataset):
    def __init__(self, annotation_file, audio_dir, sample_rate, num_samples, n_fft, hop_length, n_mels, folds):
        self.annotations = pd.read_csv(annotation_file)
        self.annotations = self.annotations[self.annotations["fold"].isin(folds)].reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.class_names = (self.annotations.sort_values("classID")["class"].unique().tolist())
    def __len__(self):
        return len(self.annotations)
    def __getitem__(self, index):
        audio_path = self._get_audio_path(index)
        label = self._get_label(index)
        signal = self._load_audio(audio_path)
        signal = self._resample(signal)
        signal = self._mix_down(signal)
        signal = self._remove_silence(signal)
        signal = self._fix_length(signal)
        mel = self._create_mel_spectrogram(signal)
        mel = self._normalize(mel)
        mel = self._to_tensor(mel)
        return mel, label
    def _get_audio_path(self, index):
        filename = self.annotations.iloc[index]["slice_file_name"]
        fold = self.annotations.iloc[index]["fold"]
        return self.audio_dir / f"fold{fold}" / filename
    def _get_label(self, index):
        return self.annotations.iloc[index]["classID"]
    def _load_audio(self, audio_path):
        signal, sr = librosa.load(audio_path, sr=None, mono=False)
        return signal, sr
    def _resample(self, data):
        signal, sr = data
        if sr != self.sample_rate:
            signal = librosa.resample(signal, orig_sr=sr, target_sr=self.sample_rate)
        return signal
    def _mix_down(self, signal):
        if signal.ndim > 1:
            signal = librosa.to_mono(signal)
        return signal
    def _remove_silence(self, signal):
        signal, _ = librosa.effects.trim(signal, top_db=20)
        return signal
    def _fix_length(self, signal):
        return librosa.util.fix_length(signal, size=self.num_samples)
    def _create_mel_spectrogram(self, signal):
        mel = librosa.feature.melspectrogram(y=signal, sr=self.sample_rate, n_fft=self.n_fft,
                                             hop_length=self.hop_length, n_mels=self.n_mels)
        mel = librosa.power_to_db(mel, ref=np.max)
        return mel
    def _normalize(self, mel):
        mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)
        return mel
    def _to_tensor(self, mel):
        mel = torch.tensor(mel, dtype=torch.float32)
        mel = mel.unsqueeze(0)
        return mel
# ----------------------------------------------------------------------------------------------------------------------
# 3)Train/Test Split:
train_dataset = UrbanSoundDataset(ANNOTATION_FILE, DATASET_PATH, SAMPLE_RATE,
                                  NUM_SAMPLES, N_FFT, HOP_LENGTH, N_MELS, [1,2,3,4,5,6,7,8,9])
test_dataset = UrbanSoundDataset(ANNOTATION_FILE, DATASET_PATH, SAMPLE_RATE,
                                  NUM_SAMPLES, N_FFT, HOP_LENGTH, N_MELS, [10])
# ----------------------------------------------------------------------------------------------------------------------
# 4)Data-Loader:
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
# ----------------------------------------------------------------------------------------------------------------------
# 5)CNN Model:
class AudioClassifierCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
                                   , nn.BatchNorm2d(16)
                                   , nn.ReLU()
                                   , nn.MaxPool2d(kernel_size=2))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
                                   , nn.BatchNorm2d(32)
                                   , nn.ReLU()
                                   , nn.MaxPool2d(kernel_size=2))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
                                   , nn.BatchNorm2d(64)
                                   , nn.ReLU()
                                   , nn.MaxPool2d(kernel_size=2))
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 8 * 5, 128)
        self.fc2 = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
model = AudioClassifierCNN(num_classes=len(train_dataset.class_names)).to(DEVICE)
# ----------------------------------------------------------------------------------------------------------------------
# 6)Loss & Optimizer:
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr= LEARNING_RATE)
scheduler = StepLR(optimizer, step_size= 5, gamma= 0.5)
# ----------------------------------------------------------------------------------------------------------------------
# 7)Training Loop:
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []
best_test_accuracy = 0
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    for mel, labels in train_loader:
        mel = mel.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(mel)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * mel.size(0)
        _, predictions = torch.max(outputs, dim=1)
        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)
    train_loss /= train_total
    train_accuracy = (train_correct/ train_total) * 100
# ----------------------------------------------------------------------------------------------------------------------
# 8)Evaluation:
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for mel, labels in test_loader:
            mel = mel.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(mel)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * mel.size(0)
            _, predictions = torch.max(outputs, 1)
            test_correct += (predictions == labels).sum().item()
            test_total += labels.size(0)
    test_loss /= test_total
    test_accuracy = 100 * test_correct / test_total
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)
    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        torch.save(model.state_dict(), "best_model.pth")
    print(f'epoch: {epoch + 1}/{EPOCHS} | LR = {optimizer.param_groups[0]["lr"]:.8f} | Train Loss = {train_loss:.4f} | '
          f'Train Accuracy = {train_accuracy:.2f}% | Test Loss = {test_loss:.4f} | Test Accuracy = {test_accuracy:.2f}%')
    scheduler.step()
history = {"train_losses": train_losses,
           "test_losses": test_losses,
           "train_accuracies": train_accuracies,
           "test_accuracies": test_accuracies}
torch.save(history, "training_history.pth")
