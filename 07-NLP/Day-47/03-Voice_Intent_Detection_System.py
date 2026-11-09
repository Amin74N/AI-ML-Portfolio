from transformers import pipeline
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import whisper
import string
import re
# ----------------------------------------------------------------------------------------------------------------------
DATA_DIR = Path("data")
AUDIO_DIR = Path("audio")
OUTPUT_DIR = Path("output")
DATASET_PATH = DATA_DIR/"intent_dataset.csv"
OUTPUT_DIR.mkdir(exist_ok=True)
# ----------------------------------------------------------------------------------------------------------------------
# 1)Loading Intent Dataset:
def load_intent_dataset(dataset):
    df = pd.read_csv(dataset)
    required_columns = {"text", "intent"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {required_columns}")
    return df
# ----------------------------------------------------------------------------------------------------------------------
# 2)Loading Audio Samples:
def load_audio_samples(audio_dir):
    audio_files = sorted(audio_dir.glob("*.wav"))
    if not audio_files:
        raise FileNotFoundError(f"No WAV files found in: {audio_dir}")
    return audio_files
def get_expected_intent(audio_path):
    return audio_path.stem.upper()
# ----------------------------------------------------------------------------------------------------------------------
# 3)Whisper ASR:
def load_whisper_model():
    model = whisper.load_model("base")
    return model
def transcribe(model, audio_path):
    result = model.transcribe(str(audio_path), fp16=False)
    return result["text"].strip()
# ----------------------------------------------------------------------------------------------------------------------
# 4)Text Cleaning:
def lowercase_text(text):
    return text.lower()
def remove_punctuations(text):
    punctuations = string.punctuation
    return text.translate(str.maketrans('', '', punctuations))
def remove_extra_whitespaces(text):
    return re.sub(r'\s+', ' ', text).strip()
def clean_text(text):
    text = lowercase_text(text)
    text = remove_punctuations(text)
    text = remove_extra_whitespaces(text)
    return text
# ----------------------------------------------------------------------------------------------------------------------
# 5)Intent Classification:
def load_model():
    classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0")
    return classifier
def classify_intent(classifier, dataset, text):
    candidate_labels = dataset["intent"].unique().tolist()
    result = classifier(text, candidate_labels=candidate_labels)
    predicted_intent = result["labels"][0]
    confidence = result["scores"][0]
    return predicted_intent, confidence, result
# ----------------------------------------------------------------------------------------------------------------------
# 6)Batch Processing:
def process_audio_files(audio_files, whisper_model, classifier, dataset):
    results = []
    for audio_path in audio_files:
        result = process_audio(audio_path, whisper_model, classifier, dataset)
        expected_intent = get_expected_intent(audio_path)
        results.append({"audio": audio_path,
                        "expected_intent": expected_intent,
                        "text": result["text"],
                        "cleaned_text": result["cleaned_text"],
                        "predicted_intent": result["intent"],
                        "confidence": result["score"]})
    return results
# ----------------------------------------------------------------------------------------------------------------------
# 7)Accuracy:
def calculate_accuracy(results):
    correct = 0
    for result in results:
        if result["predicted_intent"] == result["expected_intent"]:
            correct += 1
    total = len(results)
    accuracy = correct / total if total > 0 else 0
    evaluation_df = pd.DataFrame(results)
    evaluation_df["correct"] = (evaluation_df["predicted_intent"]== evaluation_df["expected_intent"])
    evaluation_path = OUTPUT_DIR / "evaluation.csv"
    evaluation_df.to_csv(evaluation_path, index=False)
    return correct, total, accuracy, evaluation_path
# ----------------------------------------------------------------------------------------------------------------------
# 8)Visualization:
def create_result_plot(evaluation_path, output_dir):
    evaluation_df = pd.read_csv(evaluation_path)
    correct = evaluation_df["correct"].sum()
    incorrect = len(evaluation_df) - correct
    accuracy = correct / len(evaluation_df) if len(evaluation_df) > 0 else 0
    labels = ["Correct", "Incorrect"]
    values = [correct, incorrect]
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, values)
    plt.title(f"Voice Intent Detection Results\nAccuracy: {accuracy:.2%}")
    plt.ylabel("Number of Samples")
    plt.ylim(0, len(evaluation_df))
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(value), ha="center", va="bottom")
    plt.tight_layout()
    result_path = output_dir / "result.png"
    plt.savefig(result_path, dpi=300)
    plt.close()
    return result_path
# ----------------------------------------------------------------------------------------------------------------------
# 9)Main Pipeline:
def process_audio(audio_path, whisper_model, classifier, dataset):
    text = transcribe(whisper_model, audio_path)
    cleaned_text = clean_text(text)
    predicted_intent, confidence, result = classify_intent(classifier, dataset, cleaned_text)
    if confidence < 0.50:
        predicted_intent = "NOT_RECOGNIZED"
    return {"text": text,
            "cleaned_text": cleaned_text,
            "intent": predicted_intent,
            "score": confidence}
df = load_intent_dataset(DATASET_PATH)
print("INTENT DATASET INFORMATION:\n-----------------------------------------------------------")
print("\nDataset Shape:")
print(df.shape)
print("\nDataset Columns:")
print(df.columns.tolist())
print("\nIntent Classes:")
print(df["intent"].unique().tolist())
print("\nNumber of Intent Classes:")
print(df["intent"].nunique())
print("\nMissing Values:")
print(df[["text", "intent"]].isnull().sum())
print("\nIntent Distribution:")
print(df["intent"].value_counts())
audio_files = load_audio_samples(AUDIO_DIR)
whisper_model = load_whisper_model()
classifier = load_model()
results = process_audio_files(audio_files, whisper_model, classifier, df)
print("\nBATCH PROCESSING RESULTS:\n-----------------------------------------------------------")
for result in results:
    print(f"\nAudio            : {result['audio']}")
    print(f"Expected Intent  : {result['expected_intent']}")
    print(f"Transcript       : {result['text']}")
    print(f"Cleaned Text     : {result['cleaned_text']}")
    print(f"Predicted Intent : {result['predicted_intent']}")
    print(f"Confidence       : {result['confidence']:.4f}")
correct, total, accuracy, evaluation_path = calculate_accuracy(results)
print("\nPREDICTION ACCURACY:\n-----------------------------------------------------------")
print(f"\nCorrect  : {correct}")
print(f"Total    : {total}")
print(f"Accuracy : {accuracy:.2%}")
print(f"\nEvaluation saved to: {evaluation_path}")
result_path = create_result_plot(evaluation_path, OUTPUT_DIR)
print(f"Result plot saved to: {result_path}")
