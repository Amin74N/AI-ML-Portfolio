from pathlib import Path
import whisper
import json
import time
# ----------------------------------------------------------------------------------------------------------------------
DEFAULT_MODEL = "base"
DEFAULT_LANGUAGE = None
INPUT_DIR = Path("audio")
TRANSCRIPT_FILE = Path("transcript.txt")
SEGMENTS_FILE = Path("segments.json")
# ----------------------------------------------------------------------------------------------------------------------
# Model Selection:
AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large"]
def select_model():
    print("Available Whisper Models:")
    for index, model in enumerate(AVAILABLE_MODELS, start=1):
        print(f"{index}. {model}")
    choice = input("Select model number (Default = base): ")
    if choice == "":
        return DEFAULT_MODEL
    try:
        choice = int(choice)
        if 1 <= choice <= len(AVAILABLE_MODELS):
            return AVAILABLE_MODELS[choice - 1]
        else:
            print("Invalid choice. Using default model.")
            return DEFAULT_MODEL
    except ValueError:
        print("Invalid input. Using default model.")
        return DEFAULT_MODEL
# ----------------------------------------------------------------------------------------------------------------------
# Model Loader:
def load_whisper_model(model_name):
    print(f"Loading Whisper model: {model_name}")
    start_time = time.time()
    model = whisper.load_model(model_name)
    load_time = time.time() - start_time
    print(f"Model loaded successfully in {load_time:.2f} seconds")
    return model, load_time
# ----------------------------------------------------------------------------------------------------------------------
# Audio Loader:
def load_audio():
    audio_file = INPUT_DIR / input("Enter audio file name: ")
    if not audio_file.exists():
        raise FileNotFoundError("Audio file does not exist!")
    return audio_file
# ----------------------------------------------------------------------------------------------------------------------
# Language Selection:
def select_language():
    print("Language Mode:")
    print("1. Auto Detect")
    print("2. English")
    print("3. Persian")
    choice = input("Select language mode (Default = Auto): ")
    if choice == "":
        return None
    if choice == "1":
        return None
    elif choice == "2":
        return "en"
    elif choice == "3":
        return "fa"
    else:
        print("Invalid choice. Using Auto Detection.")
        return None
# ----------------------------------------------------------------------------------------------------------------------
# Transcriber:
def transcribe_audio(model, audio, language=None):
    start_time = time.time()
    options = {}
    if language is not None:
        options["language"] = language
    result = model.transcribe(str(audio), **options, fp16=False)
    processing_time = time.time() - start_time
    print(f"Transcription completed in {processing_time:.2f} seconds")
    return result, processing_time
# ----------------------------------------------------------------------------------------------------------------------
# Language Detection:
def detect_language(result):
    language = result.get("language", None)
    if language:
        print(f"Detected Language: {language}")
    else:
        print("Language detection failed.")
    return language
# ----------------------------------------------------------------------------------------------------------------------
# Segments Extraction:
def extract_segments(result):
    segments = result.get("segments", [])
    print("Segments Information:")
    for i, segment in enumerate(segments, start=1):
        start = segment["start"]
        end = segment["end"]
        text = segment["text"].strip()
        print(f"Segment {i}: [{start:.2f}s --> {end:.2f}s]")
        print(f"{text}")
    return segments
# ----------------------------------------------------------------------------------------------------------------------
# Save Result:
def save_transcript(text, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text)
    print(f"Transcript saved: {file_path}")
def save_segments(segments, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(segments, file, ensure_ascii=False, indent=4)
    print(f"Segments saved: {file_path}")
# ----------------------------------------------------------------------------------------------------------------------
# Statistics:
def show_statistics(model_name, language, audio_path, load_time, processing_time, segments, text):
    print("Whisper Statistics :")
    print(f"Model Used        : {model_name}")
    print(f"Language          : {language}")
    print(f"Audio File        : {audio_path.name}")
    print(f"Segments          : {len(segments)}")
    print(f"Processing Time   : {processing_time:.2f} sec")
    print(f"Model Load Time   : {load_time:.2f} sec")
    print(f"Transcript Length : {len(text.split())} words")
# ----------------------------------------------------------------------------------------------------------------------
#Main:
total_start = time.time()
selected_model = select_model()
model, load_time = load_whisper_model(selected_model)
audio = load_audio()
selected_language = select_language()
result, processing_time = transcribe_audio(model, audio, selected_language)
detected_language = detect_language(result)
segments = extract_segments(result)
save_transcript(result["text"], TRANSCRIPT_FILE)
save_segments(segments, SEGMENTS_FILE)
show_statistics(selected_model, detected_language, audio, load_time, processing_time, segments, result["text"])
total_time = time.time() - total_start
print(f"Total Program Time: {total_time:.2f} sec")
