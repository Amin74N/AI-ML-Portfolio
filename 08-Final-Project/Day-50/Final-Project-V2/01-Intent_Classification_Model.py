from pathlib import Path
import re
import string
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
# ----------------------------------------------------------------------------------------------------------------------
INPUT_DIR = Path("input")
TRAIN_PATH = INPUT_DIR / "model_train.csv"
VALIDATION_PATH = INPUT_DIR / "model_validation.csv"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
BEST_MODEL_PATH = OUTPUT_DIR / "best_model.pth"
ACCURACY_CURVE_PATH = OUTPUT_DIR / "accuracy_curve.png"
LOSS_CURVE_PATH = OUTPUT_DIR / "loss_curve.png"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.png"
SEED = 42
torch.manual_seed(SEED)
EPOCHS = 10
BATCH_SIZE = 32
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ----------------------------------------------------------------------------------------------------------------------
# 1) Loading datasets:
print("[1] Loading datasets...")
print("-------------------------------------------")
train_df = pd.read_csv(TRAIN_PATH)
validation_df = pd.read_csv(VALIDATION_PATH)
print(f"\nTraining samples: {len(train_df)}")
print(f"Validation samples: {len(validation_df)}")
print("\nTraining Intent Distribution:\n", train_df["intent"].value_counts())
print("\nValidation Intent Distribution:\n", validation_df["intent"].value_counts())
required_columns = {"text", "intent"}
if not required_columns.issubset(train_df.columns):
    raise ValueError("model_train.csv must contain 'text' and 'intent' columns.")
if not required_columns.issubset(validation_df.columns):
    raise ValueError("model_validation.csv must contain 'text' and 'intent' columns.")
# ----------------------------------------------------------------------------------------------------------------------
# 2) Text preprocessing:
def clean_text(text):
    text = text.lower()
    punctuations = string.punctuation
    text = text.translate(str.maketrans("", "", punctuations))
    text = re.sub(r"\s+", " ", text).strip()
    return text
train_texts = train_df["text"].apply(clean_text).tolist()
validation_texts = validation_df["text"].apply(clean_text).tolist()
train_labels = train_df["intent"].tolist()
validation_labels = validation_df["intent"].tolist()
# ----------------------------------------------------------------------------------------------------------------------
# 3) Label Encoding:
print("\n[2] Encoding intents...")
print("-------------------------------------------")
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_labels)
y_validation = label_encoder.transform(validation_labels)
class_names = list(label_encoder.classes_)
print("\nIntent Classes:")
for index, intent in enumerate(class_names):
    print(f"{index+1}: {intent}")
NUM_CLASSES = len(class_names)
# ----------------------------------------------------------------------------------------------------------------------
# 4) TF-IDF Feature Extraction:
print("\n[3] Building TF-IDF features...")
print("-------------------------------------------")
vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, max_df=1.0, sublinear_tf=True)
X_train = vectorizer.fit_transform(train_texts)
X_validation = vectorizer.transform(validation_texts)
print(f"\nNumber of TF-IDF features: {X_train.shape[1]}")
# ----------------------------------------------------------------------------------------------------------------------
# 5) Convert to PyTorch tensors:
X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32)
X_validation_tensor = torch.tensor(X_validation.toarray(), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_validation_tensor = torch.tensor(y_validation, dtype=torch.long)
# ----------------------------------------------------------------------------------------------------------------------
# 6) Dataset / DataLoader:
train_dataset = TensorDataset( X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# ----------------------------------------------------------------------------------------------------------------------
# 7) Intent Classifier:
class IntentClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_size, hidden_size),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(hidden_size, hidden_size),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(hidden_size,num_classes))
    def forward(self, x):
        return self.network(x)
# ----------------------------------------------------------------------------------------------------------------------
# 8) Training Function:
def train_model(hidden_size, dropout, learning_rate, epochs):
    model = IntentClassifier(input_size=X_train.shape[1], hidden_size=hidden_size,
                             num_classes=NUM_CLASSES, dropout=dropout).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = []
    train_accuracies = []
    validation_losses = []
    validation_accuracies = []
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        train_correct = 0
        train_total = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            train_correct += (predictions == y_batch).sum().item()
            train_total += y_batch.size(0)
        train_loss = total_train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        model.eval()
        with torch.no_grad():
            X_val = X_validation_tensor.to(DEVICE)
            y_val = y_validation_tensor.to(DEVICE)
            validation_outputs = model(X_val)
            validation_loss = criterion(validation_outputs, y_val)
            validation_predictions = torch.argmax(validation_outputs, dim=1)
            validation_correct = (validation_predictions == y_val).sum().item()
            validation_total = y_val.size(0)
            validation_loss = validation_loss.item()
            validation_accuracy = (validation_correct / validation_total)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        validation_losses.append(validation_loss)
        validation_accuracies.append(validation_accuracy)
        print(f"Epoch [{epoch + 1:02d}/{epochs}] | Train Loss: {train_loss:.4f} | " f"Train Accuracy: {train_accuracy:.4f}"
              f" | Validation Loss: {validation_loss:.4f} | Validation Accuracy: {validation_accuracy:.4f}")
    return model, train_losses, train_accuracies, validation_losses, validation_accuracies
# ----------------------------------------------------------------------------------------------------------------------
# 9) Evaluation Function:
def evaluate_model(model):
    model.eval()
    with torch.no_grad():
        X = X_validation_tensor.to(DEVICE)
        outputs = model(X)
        predictions = torch.argmax(outputs, dim=1)
        probabilities = torch.softmax(outputs, dim=1)
        confidences = torch.max(probabilities, dim=1).values
    predictions = predictions.cpu().numpy()
    confidences = confidences.cpu().numpy()
    accuracy = accuracy_score(y_validation, predictions)
    precision = precision_score(
        y_validation,predictions, average="macro", zero_division=0)
    recall = recall_score(y_validation, predictions, average="macro", zero_division=0)
    f1 = f1_score(y_validation, predictions, average="macro", zero_division=0)
    return {"accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predictions": predictions,
            "confidences": confidences}
# ----------------------------------------------------------------------------------------------------------------------
# 10) Model Selection:
print("\n[4] Training candidate models...")
print("-------------------------------------------")
candidate_configs = [{"name": "MLP_64", "hidden_size": 64, "dropout": 0.30, "learning_rate": 0.001},
                     {"name": "MLP_128", "hidden_size": 128, "dropout": 0.30, "learning_rate": 0.001},
                     {"name": "MLP_256", "hidden_size": 256, "dropout": 0.30, "learning_rate": 0.001}]
results = []
best_model = None
best_result = None
best_config = None
best_train_losses = None
best_train_accuracies = None
best_validation_losses = None
best_validation_accuracies = None
for config in candidate_configs:
    print(f"\nTraining {config['name']}...")
    (model,train_losses, train_accuracies, validation_losses, validation_accuracies) = train_model(
        hidden_size=config["hidden_size"], dropout=config["dropout"], learning_rate=config["learning_rate"], epochs=EPOCHS)
    evaluation = evaluate_model(model)
    result = {"name": config["name"],
              "accuracy": evaluation["accuracy"],
              "precision": evaluation["precision"],
              "recall": evaluation["recall"],
              "f1": evaluation["f1"]}
    results.append(result)
    if (best_result is None or evaluation["f1"] > best_result["f1"]):
        best_model = model
        best_result = evaluation
        best_config = config
        best_train_losses = train_losses
        best_train_accuracies = train_accuracies
        best_validation_losses = validation_losses
        best_validation_accuracies = validation_accuracies
results_df = pd.DataFrame(results)
best_predictions = best_result["predictions"]
print(f"\n{results_df.to_string(index=False)}")
# ----------------------------------------------------------------------------------------------------------------------
# 11) Best Model Report:
print(f"\nBest Model: {best_config['name']}")
print(f"Accuracy  : {best_result['accuracy']:.4f}")
print(f"Precision : {best_result['precision']:.4f}")
print(f"Recall    : {best_result['recall']:.4f}")
print(f"F1-Score  : {best_result['f1']:.4f}")
report = classification_report(y_validation, best_predictions, target_names=class_names, zero_division=0)
print("\nClassification Report:")
print(report)
# ----------------------------------------------------------------------------------------------------------------------
# 12) Confusion Matrix:
cm = confusion_matrix(y_validation, best_predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Intent Classifier - Confusion Matrix")
plt.xlabel("Predicted Intent")
plt.ylabel("Expected Intent")
plt.tight_layout()
plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)
plt.close()
# ----------------------------------------------------------------------------------------------------------------------
# 13) Accuracy Curve:
plt.figure(figsize=(10, 6))
plt.plot(best_train_accuracies,label="Training Accuracy")
plt.plot(best_validation_accuracies,label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(ACCURACY_CURVE_PATH, dpi=300)
plt.close()
# ----------------------------------------------------------------------------------------------------------------------
# 14) Loss Curve:
plt.figure(figsize=(10, 6))
plt.plot(best_train_losses, label="Training Loss")
plt.plot(best_validation_losses, label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(LOSS_CURVE_PATH, dpi=300)
plt.close()
# ----------------------------------------------------------------------------------------------------------------------
# 15) Save Best Model:
print("[5] Saving best model...")
print("-------------------------------------------")
checkpoint = {"model_state_dict":best_model.state_dict(),
              "input_size":X_train.shape[1],
              "hidden_size":best_config["hidden_size"],
              "num_classes":NUM_CLASSES,
              "dropout":best_config["dropout"],
              "class_names":class_names,
              "vectorizer":vectorizer,
              "model_name":best_config["name"],
              "learning_rate":best_config["learning_rate"],
              "accuracy":best_result["accuracy"],
              "precision":best_result["precision"],
              "recall":best_result["recall"],
              "f1":best_result["f1"]}
torch.save(checkpoint, BEST_MODEL_PATH)
print("\nBest Model Saved in", BEST_MODEL_PATH)
