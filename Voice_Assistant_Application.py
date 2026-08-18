from transformers import pipeline
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import pandas as pd
import whisper
import pyaudio
import random
import string
import wave
import re
# ----------------------------------------------------------------------------------------------------------------------
INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
DATASET_PATH = INPUT_DIR / "intent_dataset.csv"
AUDIO_PATH = INPUT_DIR / "user_last_command.wav"
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_NAME = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
WHISPER_MODEL = "base"
CONFIDENCE_THRESHOLD = 0.80
FRAMES_PER_BUFFER = 3200
AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
RECORD_SECONDS = 5
# ----------------------------------------------------------------------------------------------------------------------
# 1)Loading Intent Dataset:
def load_intent_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    required_columns = {"text", "intent"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {required_columns}")
    return df
# ----------------------------------------------------------------------------------------------------------------------
# 2) Whisper ASR:
def load_whisper_model():
    model = whisper.load_model(WHISPER_MODEL)
    return model
def transcribe(model, audio_path):
    result = model.transcribe(str(audio_path), fp16=False)
    return result["text"].strip()
# ----------------------------------------------------------------------------------------------------------------------
# 3)Text Cleaning:
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
# 4)Intent Classification:
def load_classifier():
    classifier = pipeline("zero-shot-classification", model=MODEL_NAME)
    return classifier
def classify_intent(classifier, dataset, text):
    candidate_labels = dataset["intent"].unique().tolist()
    result = classifier(text, candidate_labels=candidate_labels)
    confidence = result["scores"][0]
    predicted_intent = "NOT_RECOGNIZED" if confidence < CONFIDENCE_THRESHOLD else result["labels"][0]
    return predicted_intent, confidence, result
# ----------------------------------------------------------------------------------------------------------------------
# 5)Actions:
def greeting():
    return "Hello! How can I help you?"
def get_time():
    return datetime.now().strftime("%H:%M:%S")
def get_date():
    return datetime.now().strftime("%Y-%m-%d")
def tell_joke():
    jokes = ["Why do programmers prefer dark mode? Because light attracts bugs!",
             "Why do programmers hate nature? It has too many bugs!",
             "Why did the programmer quit his job? Because he didn't get arrays!",
             "Why was the Python programmer confused? Because he couldn't C!",
             "What do you call a programmer from Finland? Nerdic!",]
    return random.choice(jokes)
def goodbye():
    return "Goodbye!"
# ----------------------------------------------------------------------------------------------------------------------
# 6)Intent Router:
ACTION_MAP = {"GREETING": greeting,
              "GET_TIME": get_time,
              "GET_DATE": get_date,
              "JOKE": tell_joke,
              "GOODBYE": goodbye}
def execute_action(intent):
    action = ACTION_MAP.get(intent)
    if action is None:
        return "I'm sorry, I didn't understand that."
    return action()
# ----------------------------------------------------------------------------------------------------------------------
# 7)Response:
def generate_response(intent, action_result):
    if intent == "GREETING":
        return action_result
    if intent == "GET_TIME":
        return f"The current time is {action_result}."
    if intent == "GET_DATE":
        return f"Today's date is {action_result}."
    if intent == "JOKE":
        return action_result
    if intent == "GOODBYE":
        return action_result
    if intent == "NOT_RECOGNIZED":
        return "I'm sorry, I didn't understand that. Please try again."
    return "I'm sorry, something went wrong."
# ----------------------------------------------------------------------------------------------------------------------
# 8)Process Audio:
def process_audio(audio_path, whisper_model, classifier, dataset):
    text = transcribe(whisper_model, audio_path)
    cleaned_text = clean_text(text)
    predicted_intent, confidence, result = classify_intent(classifier, dataset, cleaned_text)
    if predicted_intent == "NOT_RECOGNIZED":
        response = generate_response(predicted_intent, None)
    else:
        action_result = execute_action(predicted_intent)
        response = generate_response(predicted_intent, action_result)
    return {"text": text,
            "cleaned_text": cleaned_text,
            "intent": predicted_intent,
            "confidence": confidence,
            "response": response}
# ----------------------------------------------------------------------------------------------------------------------
# 9)Microphone:
def record_audio(audio_path):
    p = pyaudio.PyAudio()
    stream = p.open(format=AUDIO_FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, input=True, frames_per_buffer=FRAMES_PER_BUFFER)
    print("...")
    frames = []
    for _ in range(int(SAMPLE_RATE / FRAMES_PER_BUFFER * RECORD_SECONDS)):
        data = stream.read(FRAMES_PER_BUFFER)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    sample_width = p.get_sample_size(AUDIO_FORMAT)
    p.terminate()
    audio = wave.open(str(audio_path), "wb")
    audio.setnchannels(CHANNELS)
    audio.setsampwidth(sample_width)
    audio.setframerate(SAMPLE_RATE)
    audio.writeframes(b"".join(frames))
    audio.close()
    print("Listening finished!")
# ----------------------------------------------------------------------------------------------------------------------
# 10)Conversation Loop:
def run_assistant(dataset, whisper_model, classifier):
    session_results = []
    command_number = 1
    while True:
        print(f"\nWaiting for command {command_number}")
        record_audio(AUDIO_PATH)
        result = process_audio(AUDIO_PATH, whisper_model, classifier, dataset)
        session_results.append(result)
        print("-------------------")
        print(f"Transcript : {result['text']}")
        print(f"Intent     : {result['intent']}")
        print(f"Confidence : {result['confidence']:.4f}")
        print(f"Response   : {result['response']}")
        if result["intent"] == "GOODBYE":
            print("\nLooks like that was your last command!\n")
            break
        command_number += 1
    return session_results
# ----------------------------------------------------------------------------------------------------------------------
# 11)Session Evaluation:
def evaluate_session(session_results):
    total_commands = len(session_results)
    recognized = sum(result["intent"] != "NOT_RECOGNIZED" for result in session_results)
    not_recognized = total_commands - recognized
    recognition_rate = recognized / total_commands
    evaluation_df = pd.DataFrame(session_results)
    evaluation_df.insert(0, "command", range(1, total_commands + 1))
    evaluation_df["recognized"] = (evaluation_df["intent"] != "NOT_RECOGNIZED")
    evaluation_path = OUTPUT_DIR / "evaluation.csv"
    evaluation_df.to_csv(evaluation_path, index=False)
    print(f"Total Commands   : {total_commands}")
    print(f"Recognized       : {recognized}")
    print(f"Not Recognized   : {not_recognized}")
    print(f"Recognition Rate : {recognition_rate:.2%}")
    return (total_commands, recognized, not_recognized, recognition_rate, evaluation_path)
# ----------------------------------------------------------------------------------------------------------------------
# 12)Session Visualization:
def create_result_plot(total_commands, recognized, not_recognized, recognition_rate):
    labels = ["Recognized", "Not Recognized"]
    values = [recognized, not_recognized]
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, values)
    plt.title(f"Voice Assistant Session Results\nRecognition Rate: {recognition_rate:.2%}")
    plt.ylabel("Number of Commands")
    plt.ylim(0, total_commands)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(value), ha="center", va="bottom")
    plt.tight_layout()
    result_path = OUTPUT_DIR / "result.png"
    plt.savefig(result_path, dpi=300)
    plt.close()
# ----------------------------------------------------------------------------------------------------------------------
# 13) Main Pipeline:
print("VOICE ASSISTANT APPLICATION\n-------------------------------------------")
df = load_intent_dataset(DATASET_PATH)
whisper_model = load_whisper_model()
classifier = load_classifier()
session_results = run_assistant(df, whisper_model, classifier)
print("SESSION STATISTICS\n-------------------------------------------")
total_commands, recognized, not_recognized, recognition_rate, evaluation_path = evaluate_session(session_results)
create_result_plot(total_commands, recognized, not_recognized, recognition_rate)
