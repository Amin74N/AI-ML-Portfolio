from pathlib import Path
import speech_recognition as sr
# ----------------------------------------------------------------------------------------------------------------------
INPUT_DIR = Path("audio")
TRANSCRIPT_FILE = Path("transcript.txt")
# ----------------------------------------------------------------------------------------------------------------------
# Loading Audio:
def load_audio():
    audio_file = INPUT_DIR / input("Enter audio file name: ")
    if not audio_file.exists():
        raise FileNotFoundError("Audio file does not exist!")
    return audio_file
# ----------------------------------------------------------------------------------------------------------------------
# Language Selection:
def select_language():
    print("Select language:")
    print("1. English")
    print("2. Persian")
    choice = input("Choose (1/2): ")
    languages = {"1": "en-US",
                 "2": "fa-IR"}
    if choice not in languages:
        raise ValueError("Invalid language selection!")
    return languages[choice]
# ----------------------------------------------------------------------------------------------------------------------
# Speech To Text:
def speech_to_text(audio_file, language):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(str(audio_file)) as source:
            audio_data = recognizer.record(source)
        transcript = recognizer.recognize_google(audio_data, language=language)
        return transcript
    except ValueError:
        return "Invalid audio data."
    except OSError:
        return "Unsupported or corrupted audio file."
    except sr.UnknownValueError:
        return "Could not understand the audio."
    except sr.RequestError:
        return "Speech recognition service unavailable."
# ----------------------------------------------------------------------------------------------------------------------
# Saving Transcript:
def save_transcript(transcript):
    TRANSCRIPT_FILE.write_text(transcript, encoding="utf-8")
    print(f"Transcript saved to: {TRANSCRIPT_FILE}")
# ----------------------------------------------------------------------------------------------------------------------
# Main:
try:
    audio = load_audio()
    language = select_language()
    transcript = speech_to_text(audio, language)
    print("Transcript:")
    print(transcript)
    save_transcript(transcript)
except FileNotFoundError as error:
    print(f"Error: {error}")
except ValueError as error:
    print(f"Error: {error}")
