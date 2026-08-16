from pathlib import Path
import pandas as pd
from transformers import pipeline
# ----------------------------------------------------------------------------------------------------------------------
INPUTS_DIR = Path("inputs")
INTENTS_FILE = INPUTS_DIR / "intents.csv"
AUDIO_FILE = INPUTS_DIR / "sample.wav"
RESPONSES = {"GREETING": "Hello! How can I help you?",
             "ORDER_STATUS": "Your order is being processed.",
             "COMPLAINT": "I'm sorry to hear that. We will investigate your complaint.",
             "QUESTION": "Sure! I will try to answer your question.",
             "GOODBYE": "Goodbye! Have a nice day.",}
# ----------------------------------------------------------------------------------------------------------------------
# 1)Input Validation:
def validate_inputs():
    if not INPUTS_DIR.exists():
        raise FileNotFoundError(f"Inputs directory not found: {INPUTS_DIR}")
    if not INTENTS_FILE.exists():
        raise FileNotFoundError(f"Intent dataset not found: {INTENTS_FILE}")
    if not AUDIO_FILE.exists():
        raise FileNotFoundError(f"Audio file not found: {AUDIO_FILE}")
# ----------------------------------------------------------------------------------------------------------------------
# 2)Load Intent Dataset:
def load_intent_dataset():
    df = pd.read_csv(INTENTS_FILE)
    required_columns = {"text", "intent"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {required_columns}")
    return df
# ----------------------------------------------------------------------------------------------------------------------
# 3)Build Intent Descriptions from Dataset:
def build_intent_descriptions(df):
    intent_descriptions = {}
    for intent in df["intent"].unique():
        examples = df[df["intent"] == intent]["text"].tolist()
        intent_descriptions[intent] = (f"{intent}: " + "; ".join(examples))
    return intent_descriptions
# ----------------------------------------------------------------------------------------------------------------------
# 4)Load Hugging Face Models:
def load_models():
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-base")
    return classifier, asr
# ----------------------------------------------------------------------------------------------------------------------
# 5)Speech Recognition:
def transcribe_audio(asr):
    result = asr(str(AUDIO_FILE))
    transcript = result["text"].strip()
    return transcript
# ----------------------------------------------------------------------------------------------------------------------
# 6)Intent Classification:
def classify_intent(classifier, intent_descriptions, transcript):
    candidate_labels = list(intent_descriptions.values())
    result = classifier(transcript, candidate_labels=candidate_labels)
    predicted_description = result["labels"][0]
    confidence = result["scores"][0]
    predicted_intent = predicted_description.split(":", 1)[0]
    return predicted_intent, confidence, result
# ----------------------------------------------------------------------------------------------------------------------
# 7)Generate Response:
def generate_response(intent):
    return RESPONSES.get(intent, "I'm sorry, I don't understand your request.")
# ----------------------------------------------------------------------------------------------------------------------
# 8)Display Dataset Information:
def display_dataset_info(df):
    print("Intent Dataset:")
    print(df)
    print("\nIntent Labels:")
    print(df["intent"].unique().tolist())
# ----------------------------------------------------------------------------------------------------------------------
# 9)Display Classification Results:
def display_classification_results(result):
    print("\nZero-Shot Result:")
    for label, score in zip(result["labels"], result["scores"]):
        print(f"{label}: {score:.4f}")
# ----------------------------------------------------------------------------------------------------------------------
# 10)Display Final Result:
def display_final_result(transcript, intent, confidence, response):
    print("\nFINAL RESULT:")
    print(f"Transcript:\n{transcript}")
    print(f"Intent:\n{intent}")
    print(f"Confidence:\n{confidence:.2f}")
    print(f"Response:\n{response}")
# ----------------------------------------------------------------------------------------------------------------------
# 11)Main Pipeline:
validate_inputs()
df = load_intent_dataset()
display_dataset_info(df)
intent_descriptions = build_intent_descriptions(df)
classifier, asr = load_models()
transcript = transcribe_audio(asr)
intent, confidence, result = classify_intent(classifier, intent_descriptions, transcript)
response = generate_response(intent)
display_classification_results(result)
display_final_result(transcript, intent, confidence, response)
