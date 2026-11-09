from pathlib import Path
from datetime import datetime
import random
import string
import wave
import re
import logging
import time
import tempfile
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import whisper
import pyaudio
from gtts import gTTS
from playsound import playsound
import winsound
# ----------------------------------------------------------------------------------------------------------------------
INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
MUSIC_DIR = Path("music")
BEST_MODEL_PATH = OUTPUT_DIR / "best_model.pth"
ASSISTANT_TEST_PATH = INPUT_DIR / "assistant_test.csv"
AUDIO_PATH = INPUT_DIR / "user_last_command.wav"
LOG_PATH = OUTPUT_DIR / "assistant.log"
EVALUATION_PATH = OUTPUT_DIR / "assistant_evaluation.csv"
RESULT_PATH = OUTPUT_DIR / "assistant_result.png"
OUTPUT_DIR.mkdir(exist_ok=True)
INPUT_DIR.mkdir(exist_ok=True)
WHISPER_MODEL = "small"
CONFIDENCE_THRESHOLD = 0.75
FRAMES_PER_BUFFER = 3200
AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
RECORD_SECONDS = 5
SILENCE_THRESHOLD = 500
MIN_SPEECH_DURATION = 0.15
SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(filename=LOG_PATH, level=logging.INFO, format="%(asctime)s | %(message)s")
LOGGER = logging.getLogger(__name__)
# ----------------------------------------------------------------------------------------------------------------------
# 1)INTENT CLASSIFIER:
class IntentClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_size, hidden_size),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(hidden_size, hidden_size),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(hidden_size, num_classes))
    def forward(self, x):
        return self.network(x)
# ----------------------------------------------------------------------------------------------------------------------
# 2)LOAD ASSISTANT TEST DATASET:
def load_assistant_test_dataset():
    if not ASSISTANT_TEST_PATH.exists():
        raise FileNotFoundError(f"assistant_test.csv not found: {ASSISTANT_TEST_PATH}")
    df = pd.read_csv(ASSISTANT_TEST_PATH, keep_default_na=False)
    required_columns = {"text", "intent"}
    if not required_columns.issubset(df.columns):
        raise ValueError("assistant_test.csv must contain 'text' and 'intent' columns.")
    df["text"] = df["text"].fillna("").astype(str)
    df["intent"] = df["intent"].fillna("").astype(str)
    df = df.reset_index(drop=True)
    print("Test dataset loaded successfully.")
    return df
# ----------------------------------------------------------------------------------------------------------------------
# 3)LOAD BEST INTENT CLASSIFICATION MODEL:
def load_intent_classifier():
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Best model not found: {BEST_MODEL_PATH}")
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=False)
    input_size = checkpoint["input_size"]
    hidden_size = checkpoint["hidden_size"]
    num_classes = checkpoint["num_classes"]
    dropout = checkpoint["dropout"]
    class_names = checkpoint["class_names"]
    vectorizer = checkpoint["vectorizer"]
    model_name = checkpoint["model_name"]
    model = IntentClassifier(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes, dropout=dropout)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    print("Classification Model loaded successfully.")
    return model, vectorizer, class_names
# ----------------------------------------------------------------------------------------------------------------------
# 4)WHISPER ASR:
def load_whisper_model():
    model = whisper.load_model(WHISPER_MODEL)
    print("Whisper model loaded successfully.")
    return model
def transcribe(whisper_model, audio_path):
    try:
        result = whisper_model.transcribe(str(audio_path), fp16=False)
        text = result["text"].strip()
        return text
    except Exception as error:
        raise RuntimeError(f"ASR Error: {error}")
# ----------------------------------------------------------------------------------------------------------------------
# 5)TEXT CLEANING:
def lowercase_text(text):
    return text.lower()
def remove_punctuations(text):
    punctuations = string.punctuation
    return text.translate(str.maketrans("", "", punctuations))
def remove_extra_whitespaces(text):
    return re.sub(r"\s+", " ", text).strip()
def clean_text(text):
    text = lowercase_text(text)
    text = remove_punctuations(text)
    text = remove_extra_whitespaces(text)
    return text
# ----------------------------------------------------------------------------------------------------------------------
# 6)AUDIO PROCESSING:
def normalize_audio(audio_data):
    audio = np.asarray(audio_data, dtype=np.float32)
    if len(audio) == 0:
        raise ValueError("Empty audio!")
    max_amplitude = np.max(np.abs(audio))
    if max_amplitude == 0:
        return audio.astype(np.int16)
    normalized = (audio / max_amplitude) * 32767
    return normalized.astype(np.int16)
def remove_silence(audio_data):
    audio = np.asarray(audio_data, dtype=np.int16)
    if len(audio) == 0:
        raise ValueError("Empty audio!")
    amplitude = np.abs(audio.astype(np.int32))
    active_indices = np.where(amplitude > SILENCE_THRESHOLD)[0]
    if len(active_indices) == 0:
        return np.array([], dtype=np.int16)
    start = active_indices[0]
    end = active_indices[-1] + 1
    speech_duration = (end - start) / SAMPLE_RATE
    if speech_duration < MIN_SPEECH_DURATION:
        return np.array([], dtype=np.int16)
    return audio[start:end]
def process_audio_file(audio_path):
    try:
        with wave.open(str(audio_path), "rb") as audio:
            frames = audio.readframes(audio.getnframes())
            sample_width = audio.getsampwidth()
            frame_rate = audio.getframerate()
            channels = audio.getnchannels()
        audio_data = np.frombuffer(frames, dtype=np.int16)
        if len(audio_data) == 0:
            raise ValueError("Empty audio!")
        audio_data = remove_silence(audio_data)
        if len(audio_data) == 0:
            return False
        audio_data = normalize_audio(audio_data)
        with wave.open(str(audio_path), "wb") as processed_audio:
            processed_audio.setnchannels(channels)
            processed_audio.setsampwidth(sample_width)
            processed_audio.setframerate(frame_rate)
            processed_audio.writeframes(audio_data.tobytes())
        return True
    except Exception as error:
        raise RuntimeError(f"Audio Processing Error: {error}")
# ----------------------------------------------------------------------------------------------------------------------
# 7)INTENT CLASSIFICATION:
def classify_intent(model, vectorizer, class_names, text):
    if not text.strip():
        return ("NOT_RECOGNIZED", 0.0)
    features = vectorizer.transform([text])
    features = torch.tensor(features.toarray(), dtype=torch.float32).to(DEVICE)
    model.eval()
    with torch.no_grad():
        outputs = model(features)
        probabilities = torch.softmax(outputs, dim=1)
        confidence_tensor = torch.max(probabilities, dim=1)
        confidence = confidence_tensor.values.item()
        predicted_index = confidence_tensor.indices.item()
    predicted_intent = class_names[predicted_index]
    if confidence < CONFIDENCE_THRESHOLD:
        predicted_intent = "NOT_RECOGNIZED"
    return (predicted_intent, confidence)
# ----------------------------------------------------------------------------------------------------------------------
# 8)ACTIONS:
def greeting():
    return "Hello! How can I help you?"
def get_time():
    return datetime.now().strftime("%H:%M:%S")
def get_date():
    return datetime.now().strftime("%Y-%m-%d")
def set_timer(command_text):
    numbers = re.findall(r"\b\d+\b", command_text)
    if not numbers:
        return "I couldn't determine the timer duration."
    seconds = int(numbers[0])
    message = (f"Timer set for {seconds} seconds.")
    print(f"\n{message}")
    time.sleep(seconds)
    if seconds == 1:
        return "Your 1 second timer finished."
    return (f"Your {seconds} second timer finished.")
def play_music():
    if not MUSIC_DIR.exists():
        return "The music folder was not found."
    music_files = list(MUSIC_DIR.glob("*.wav"))
    if not music_files:
        return "I couldn't find any music files."
    selected_music = random.choice(music_files)
    print(f"\nPlaying {selected_music.name}")
    try:
        winsound.PlaySound(str(selected_music), winsound.SND_FILENAME)
        return (f"{selected_music.stem} was played.")
    except Exception as error:
        LOGGER.error(f"Music playback error | {error}")
        return (f"I selected {selected_music.stem}, but I couldn't play it.")
def get_weather():
    return ("I don't have live weather information right now.")
def set_reminder():
    reminders = ["Remember to stay focused on what matters.",
                 "Remember to take things one step at a time.",
                 "Remember to learn as much as possible.",
                 "Remember that consistency is more important than perfection.",
                 "Remember to keep moving forward."]
    return random.choice(reminders)
def tell_joke():
    jokes = ["Why do programmers prefer dark mode? Because light attracts bugs!",
             "Why do programmers hate nature? It has too many bugs!",
             "Why did the programmer quit his job? Because he didn't get arrays!",
             "Why was the Python programmer confused? Because he couldn't C!",
             "What do you call a programmer from Finland? Nerdic!"]
    return random.choice(jokes)
def motivation():
    messages = ["Keep going. Small progress is still progress.",
                "You are closer than you think. Don't give up.",
                "Stay consistent and trust the process.",
                "You don't need to be perfect. You just need to keep moving.",
                "Your future self will thank you for not giving up today."]
    return random.choice(messages)
def goodbye():
    return ("It was a good session. Goodbye!")
def not_recognized():
    return ("I'm sorry, I didn't understand that. Please try again.")
# ----------------------------------------------------------------------------------------------------------------------
# 9)INTENT ROUTER:
def execute_action(intent, command_text):
    if intent == "GREETING":
        return greeting()
    if intent == "GET_TIME":
        return get_time()
    if intent == "GET_DATE":
        return get_date()
    if intent == "SET_TIMER":
        return set_timer(command_text)
    if intent == "PLAY_MUSIC":
        return play_music()
    if intent == "GET_WEATHER":
        return get_weather()
    if intent == "SET_REMINDER":
        return set_reminder()
    if intent == "TELL_JOKE":
        return tell_joke()
    if intent == "MOTIVATION":
        return motivation()
    if intent == "GOODBYE":
        return goodbye()
    if intent == "NOT_RECOGNIZED":
        return not_recognized()
    raise ValueError(f"Unknown Intent: {intent}")
# ----------------------------------------------------------------------------------------------------------------------
# 10)RESPONSE GENERATION:
def generate_response(intent, action_result):
    if intent == "GREETING":
        return action_result
    if intent == "GET_TIME":
        return (f"The current time is {action_result}.")
    if intent == "GET_DATE":
        return (f"Today's date is {action_result}.")
    if intent == "SET_TIMER":
        return action_result
    if intent == "PLAY_MUSIC":
        return action_result
    if intent == "GET_WEATHER":
        return action_result
    if intent == "SET_REMINDER":
        return action_result
    if intent == "TELL_JOKE":
        return action_result
    if intent == "MOTIVATION":
        return action_result
    if intent == "GOODBYE":
        return action_result
    if intent == "NOT_RECOGNIZED":
        return action_result
    return ("I'm sorry, something went wrong.")
# ----------------------------------------------------------------------------------------------------------------------
# 11)GOOGLE TTS:
def speak_response(response):
    temporary_file = None
    try:
        temporary_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        temporary_path = temporary_file.name
        temporary_file.close()
        tts = gTTS(text=response, lang="en")
        tts.save(temporary_path)
        playsound(temporary_path)
    except Exception as error:
        raise RuntimeError(f"TTS Error: {error}")
    finally:
        if temporary_file is not None:
            try:
                if os.path.exists(temporary_file.name):
                    os.remove(temporary_file.name)
            except Exception:
                pass
# ----------------------------------------------------------------------------------------------------------------------
# 12)MICROPHONE:
def record_audio(audio_path):
    p = pyaudio.PyAudio()
    stream = None
    try:
        stream = p.open(format=AUDIO_FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, input=True, frames_per_buffer=FRAMES_PER_BUFFER)
        print("\nRecording started.\n...")
        frames = []
        for _ in range(int(SAMPLE_RATE / FRAMES_PER_BUFFER * RECORD_SECONDS)):
            data = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
            frames.append(data)
        print("Recording finished.")
        sample_width = p.get_sample_size(AUDIO_FORMAT)
        with wave.open(str(audio_path), "wb") as audio:
            audio.setnchannels(CHANNELS)
            audio.setsampwidth(sample_width)
            audio.setframerate(SAMPLE_RATE)
            audio.writeframes(b"".join(frames))
    except Exception as error:
        raise RuntimeError(f"Microphone Error: {error}")
    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        p.terminate()
# ----------------------------------------------------------------------------------------------------------------------
# 13)PROCESS ONE COMMAND:
def process_command(command_number, expected_intent, model, vectorizer, class_names, whisper_model):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result = {"Command Number": command_number,
              "Timestamp": timestamp,
              "Expected Intent": expected_intent,
              "Transcript": "",
              "Cleaned Text": "",
              "Predicted Intent": "",
              "Confidence": 0.0,
              "Recognition": "Not Recognized",
              "Accuracy": "Incorrect",
              "Action": "",
              "Response": ""}
    try:
        record_audio(AUDIO_PATH)
    except Exception as error:
        print(f"\nERROR: {error}")
        LOGGER.error(f"Command #{command_number} | Microphone Error | {error}")
        result["Action"] = "Microphone Error"
        result["Response"] = "Microphone error."
        result["Predicted Intent"] = "NOT_RECOGNIZED"
        result["Accuracy"] = ("Correct" if expected_intent == "NOT_RECOGNIZED" else "Incorrect")
        return result
    try:
        audio_has_speech = process_audio_file(AUDIO_PATH)
        if not audio_has_speech:
            result["Transcript"] = ""
            result["Cleaned Text"] = ""
            result["Predicted Intent"] = "NOT_RECOGNIZED"
            result["Confidence"] = 0.0
            result["Recognition"] = "Not Recognized"
            result["Accuracy"] = ("Correct" if expected_intent == "NOT_RECOGNIZED" else "Incorrect")
            result["Action"] = "NOT_RECOGNIZED"
            result["Response"] = ""
            print(f"\nTranscript  : ")
            print(f"Predicted   : NOT_RECOGNIZED")
            print(f"Confidence  : 0.00%")
            print(f"Recognition : Not Recognized")
            print(f"Expected    : {result['Expected Intent']}")
            print(f"Accuracy    : {result['Accuracy']}")
            print(f"Response    : ")
            LOGGER.info(f"Command #{command_number} | "
                        f"Silence Detected | "
                        f"Expected Intent: {result['Expected Intent']} | "
                        f"Predicted Intent: NOT_RECOGNIZED | "
                        f"Confidence: 0.0000 | "
                        f"Recognition: Not Recognized | "
                        f"Accuracy: {result['Accuracy']}")
            return result
    except Exception as error:
        print(f"\nERROR: {error}")
        LOGGER.error(f"Command #{command_number} | Audio Processing Error | {error}")
        result["Action"] = "Audio Processing Error"
        result["Response"] = "Audio processing error."
        result["Predicted Intent"] = "NOT_RECOGNIZED"
        result["Accuracy"] = ("Correct" if expected_intent == "NOT_RECOGNIZED" else "Incorrect")
        return result
    try:
        transcript = transcribe(whisper_model, AUDIO_PATH)
        result["Transcript"] = transcript
    except Exception as error:
        print(f"\nERROR: {error}")
        LOGGER.error(f"Command #{command_number} | ASR Error | {error}")
        result["Action"] = "ASR Error"
        result["Response"] = ("Speech recognition error.")
        result["Predicted Intent"] = ("NOT_RECOGNIZED")
        result["Accuracy"] = ("Correct" if expected_intent == "NOT_RECOGNIZED" else "Incorrect")
        return result
    cleaned_text = clean_text(transcript)
    result["Cleaned Text"] = cleaned_text
    try:
        predicted_intent, confidence = classify_intent(model, vectorizer, class_names, cleaned_text)
        result["Predicted Intent"] = (predicted_intent)
        result["Confidence"] = (confidence)
    except Exception as error:
        print(f"\nERROR: {error}")
        LOGGER.error(f"Command #{command_number} | Classification Error | {error}")
        result["Action"] = ("Classification Error")
        result["Response"] = ("Intent classification error.")
        result["Predicted Intent"] = ("NOT_RECOGNIZED")
        result["Confidence"] = 0.0
    if result["Predicted Intent"] == "NOT_RECOGNIZED":
        result["Recognition"] = ("Not Recognized")
    else:
        result["Recognition"] = ("Recognized")
    if (result["Predicted Intent"] == expected_intent):
        result["Accuracy"] = "Correct"
    else:
        result["Accuracy"] = "Incorrect"
    try:
        action_result = execute_action(result["Predicted Intent"], cleaned_text)
        result["Action"] = (result["Predicted Intent"])
        response = generate_response(result["Predicted Intent"], action_result)
        result["Response"] = response
    except Exception as error:
        print(f"\nERROR: {error}")
        LOGGER.error(f"Command #{command_number} | Action Error | {error}")
        result["Action"] = ("Unknown Intent")
        result["Response"] = ("I'm sorry, something went wrong.")
    print(f"\nTranscript  : {result['Transcript']}")
    print(f"Predicted   : {result['Predicted Intent']}")
    print(f"Confidence  : {result['Confidence']:.2%}")
    print(f"Recognition : {result['Recognition']}")
    print(f"Expected    : {result['Expected Intent']}")
    print(f"Accuracy    : {result['Accuracy']}")
    print(f"Response    : {result['Response']}")
    try:
        speak_response(result["Response"])
    except Exception as error:
        print(f"\nTTS ERROR: {error}")
        LOGGER.error(f"Command #{command_number} | TTS Error | {error}")
    LOGGER.info(f"Command #{command_number} | "
                f"Expected Intent: {result['Expected Intent']} | "
                f"Transcript: {result['Transcript']} | "
                f"Cleaned Text: {result['Cleaned Text']} | "
                f"Predicted Intent: {result['Predicted Intent']} | "
                f"Confidence: {result['Confidence']:.4f} | "
                f"Recognition: {result['Recognition']} | "
                f"Accuracy: {result['Accuracy']} | "
                f"Action: {result['Action']} | "
                f"Response: {result['Response']}")
    return result
# ----------------------------------------------------------------------------------------------------------------------
# 14)EVALUATION LOOP:
def run_evaluation(assistant_test_df, model, vectorizer, class_names, whisper_model):
    session_results = []
    total_tests = len(assistant_test_df)
    print("\nVOICE ASSISTANT APPLICATION")
    print("-------------------------------------------")
    for index, row in assistant_test_df.iterrows():
        command_number = index + 1
        expected_intent = row["intent"]
        print(f"\nCommand {command_number}:")
        input("Press ENTER and speak for 5 seconds...")
        result = process_command(command_number, expected_intent, model, vectorizer, class_names, whisper_model)
        session_results.append(result)
    return session_results
# ----------------------------------------------------------------------------------------------------------------------
# 15)SESSION EVALUATION:
def evaluate_session(session_results):
    evaluation_df = pd.DataFrame(session_results)
    total_commands = len(evaluation_df)
    recognized = (evaluation_df["Recognition"] == "Recognized").sum()
    not_recognized = (evaluation_df["Recognition"] == "Not Recognized").sum()
    correct = (evaluation_df["Accuracy"] == "Correct").sum()
    incorrect = (evaluation_df["Accuracy"] == "Incorrect").sum()
    recognition_rate = (recognized / total_commands if total_commands > 0 else 0)
    accuracy = (correct / total_commands if total_commands > 0 else 0)
    evaluation_df.to_csv(EVALUATION_PATH, index=False)
    print("\nSESSION STATISTICS")
    print("-------------------------------------------")
    print(f"Total Commands        : {total_commands}")
    print(f"Recognized            : {recognized}")
    print(f"Not Recognized        : {not_recognized}")
    print(f"Correct Predictions   : {correct}")
    print(f"Incorrect Predictions : {incorrect}")
    print(f"Recognition Rate      : {recognition_rate:.2%}")
    print(f"Accuracy              : {accuracy:.2%}")
    return (evaluation_df, total_commands, recognized, not_recognized, correct, incorrect, recognition_rate, accuracy)
# ----------------------------------------------------------------------------------------------------------------------
# 16)RESULT VISUALIZATION:
def create_result_plot(total_commands, recognized, not_recognized, correct, incorrect, recognition_rate, accuracy):
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    recognition_labels = ["Recognized", "Not Recognized"]
    recognition_values = [recognized, not_recognized]
    bars = axes[0].bar(recognition_labels, recognition_values)
    axes[0].set_title("Recognition Results")
    axes[0].set_ylabel("Number of Commands")
    axes[0].set_ylim(0, max(1, total_commands * 1.15))
    for bar, value in zip(bars, recognition_values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(value), ha="center", va="bottom")
    axes[0].text(0.5, 0.95, f"Recognition Rate\n{recognition_rate:.2%}", transform=axes[0].transAxes,
                 ha="center", va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    axes[0].grid(axis="y", alpha=0.3)
    accuracy_labels = ["Correct", "Incorrect"]
    accuracy_values = [correct, incorrect]
    bars = axes[1].bar(accuracy_labels, accuracy_values)
    axes[1].set_title("Prediction Accuracy")
    axes[1].set_ylabel("Number of Commands")
    axes[1].set_ylim(0, max(1, total_commands * 1.15))
    for bar, value in zip(bars, accuracy_values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(value), ha="center", va="bottom")
    axes[1].text(0.5, 0.95, f"Accuracy\n{accuracy:.2%}", transform=axes[1].transAxes, ha="center",
                 va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    axes[1].grid(axis="y", alpha=0.3)
    figure.suptitle("Voice Assistant Evaluation Results",fontsize=16)
    figure.text(0.5, 0.01, f"Total Test Cases: {total_commands}", ha="center")
    plt.tight_layout(rect=[0, 0.04, 1, 0.94])
    plt.savefig(RESULT_PATH, dpi=300)
    plt.close()
 # ----------------------------------------------------------------------------------------------------------------------
# 17)MAIN:
def main():
    try:
        assistant_test_df = (load_assistant_test_dataset())
        classifier, vectorizer, class_names = (load_intent_classifier())
        whisper_model = (load_whisper_model())
        print("Everything is ready. Let's go!")
        session_results = run_evaluation(assistant_test_df, classifier, vectorizer, class_names, whisper_model)
        (evaluation_df, total_commands, recognized, not_recognized, correct, incorrect, recognition_rate,
         accuracy) = evaluate_session(session_results)
        create_result_plot(total_commands, recognized, not_recognized, correct, incorrect, recognition_rate, accuracy)
        print(f"Evaluation saved to   : {EVALUATION_PATH}")
        print(f"Result plot saved to  : {RESULT_PATH}")
        print(f"Session Logs saved to : {LOG_PATH}")
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
        LOGGER.warning("Evaluation interrupted by user.")
    except Exception as error:
        print(f"\nFATAL ERROR: {error}")
        LOGGER.critical(f"Fatal Error | {error}")
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
