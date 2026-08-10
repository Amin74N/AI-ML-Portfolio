from pathlib import Path
import librosa
import whisper
import time
# ----------------------------------------------------------------------------------------------------------------------
SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
TARGET_SAMPLE_RATE = 16000
MODEL = whisper.load_model("base")
AUDIO_DIR = Path("audio")
TRANSCRIPT_FILE = Path("transcripts.txt")
REPORT_FILE = Path("performance_report.txt")
# ----------------------------------------------------------------------------------------------------------------------
# 1)Audio Loading:
def load_audio(file_path):
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")
    if file_path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported audio format: {file_path.suffix}")
    audio, sample_rate = librosa.load(file_path, sr=None, mono=False)
    channels = 1 if audio.ndim == 1 else audio.shape[0]
    num_samples = audio.shape[0] if channels == 1 else audio.shape[1]
    return audio, sample_rate, channels, num_samples
# ----------------------------------------------------------------------------------------------------------------------
# 2)Audio Preprocessing:
def preprocess_audio(audio, sample_rate, channels):
    if channels > 1:
        audio = librosa.to_mono(audio)
        channels = 1
    if sample_rate != TARGET_SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE)
    sample_rate = TARGET_SAMPLE_RATE
    max_value = abs(audio).max()
    if max_value > 0:
        audio = audio / max_value
    audio, _ = librosa.effects.trim(audio, top_db=20)
    return audio, sample_rate, channels
# ----------------------------------------------------------------------------------------------------------------------
# 3)ASR Module:
def transcribe_audio(model, audio):
    result = model.transcribe(audio, fp16=False)
    return result
# ----------------------------------------------------------------------------------------------------------------------
# 4)Text Processing:
def process_text(text):
    if not text:
        return ""
    text = " ".join(text.split())
    text = text.strip()
    return text
# ----------------------------------------------------------------------------------------------------------------------
# 5)Transcript Formatting:
def format_transcript(file_name,language, text):
    output = (f"File Name : {file_name}\n"
              f"Language  : {language}\n"
              f"Transcript: {text}\n")
    return output
# ----------------------------------------------------------------------------------------------------------------------
# 6)File Pipeline:
def process_file(audio_path, model):
    start_time = time.perf_counter()
    audio, sample_rate, channels, num_samples = load_audio(audio_path)
    duration = num_samples / sample_rate
    processed_audio, processed_sample_rate, processed_channels = preprocess_audio(audio, sample_rate, channels)
    result = transcribe_audio(model,processed_audio)
    clean_text = process_text(result["text"])
    processing_time = time.perf_counter() - start_time
    rtf = processing_time / duration if duration > 0 else 0
    transcript = format_transcript(audio_path.name, result["language"], clean_text)
    report = format_report(audio_path.name, result["language"], duration, processing_time, rtf)
    return transcript, report
# ----------------------------------------------------------------------------------------------------------------------
# 7)Batch Processing:
def process_directory(audio_dir, model):
    audio_dir = Path(audio_dir)
    TRANSCRIPT_FILE.write_text("", encoding="utf-8")
    REPORT_FILE.write_text("", encoding="utf-8")
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    if not audio_dir.is_dir():
        raise ValueError(f"Path is not a directory: {audio_dir}")
    audio_files = sorted(file for file in audio_dir.iterdir() if file.is_file()
                         and file.suffix.lower() in SUPPORTED_FORMATS)
    if not audio_files:
        print("No supported audio files found.")
        return
    for audio_file in audio_files:
        try:
            transcript, report = process_file(audio_file, model)
            save_transcript(transcript, TRANSCRIPT_FILE)
            save_report(report, REPORT_FILE)
            print(transcript)
        except Exception as error:
            error_report = format_error_report(audio_file.name, error)
            save_report(error_report, REPORT_FILE)
            print(f"Error processing {audio_file.name}: {error}")
# ----------------------------------------------------------------------------------------------------------------------
# 8)Output Generating:
def save_transcript(transcript, output_file):
    with output_file.open("a", encoding="utf-8") as file:
        file.write(transcript)
        file.write("\n")
# ----------------------------------------------------------------------------------------------------------------------
# 9)Performance Reporting:
def format_report(file_name, language, duration, processing_time, rtf):
    output = (f"File Name       : {file_name}\n"
              f"Status          : SUCCESS\n"
              f"Language        : {language}\n"
              f"Duration        : {duration:.2f} sec\n"
              f"Processing Time : {processing_time:.2f} sec\n"
              f"Real-Time Factor: {rtf:.2f}\n\n")
    return output
# ----------------------------------------------------------------------------------------------------------------------
# 10)Report Saving:
def save_report(report, output_file):
    with output_file.open("a", encoding="utf-8") as file:
        file.write(report)
# ----------------------------------------------------------------------------------------------------------------------
# 11)Failure Reporting
def format_error_report(file_name, error):
    output = (f"File Name       : {file_name}\n"
              f"Status          : FAILED\n"
              f"Error           : {error}\n\n")
    return output
# ----------------------------------------------------------------------------------------------------------------------
# 12)Main:
process_directory(AUDIO_DIR, MODEL)
