from pathlib import Path
import time
import numpy as np
import torch
import librosa
import soundfile as sf
# ----------------------------------------------------------------------------------------------------------------------
AUDIO_DIR = Path("audio")
AUDIO_FILE = AUDIO_DIR / "English.wav"
PROCESSED_AUDIO = AUDIO_DIR / "English_16k.wav"
OUTPUT_FILE = Path("transcripts.txt")
# ----------------------------------------------------------------------------------------------------------------------
# 1)Audio preprocessing:
def preprocess_audio(input_path, output_path):
    audio, sample_rate = sf.read(input_path)
    print("Original Audio")
    print("Sample Rate:", sample_rate)
    print("Shape:", audio.shape)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sample_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
    sf.write(output_path, audio, 16000)
    print("Preprocessing completed")
    print("New Sample Rate: 16000")
    print("Samples:", len(audio))
    print("Duration:", len(audio)/16000, "seconds")
    return output_path
# ----------------------------------------------------------------------------------------------------------------------
# 2)Save results:
def save_result(model_name, runtime, transcript):
    with open(OUTPUT_FILE, "a", encoding="utf-8") as file:
        file.write(f"{model_name} \n")
        file.write(f"Runtime: {runtime:.2f} seconds\n")
        file.write(f"Transcript:\n {transcript} \n")
        file.write("------------------------------------------------------------------------------------------------\n")
# ----------------------------------------------------------------------------------------------------------------------
# 3)Whisper:
def run_whisper(audio_path):
    import whisper
    model = whisper.load_model("base")
    start = time.time()
    result = model.transcribe(str(audio_path))
    runtime = time.time() - start
    transcript = result["text"].strip()
    save_result("Whisper_result", runtime, transcript)
    return transcript, runtime
# ----------------------------------------------------------------------------------------------------------------------
# 4)Wav2Vec2:
def run_wav2vec2(audio_path):
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
    start = time.time()
    result = asr(str(audio_path))
    runtime = time.time() - start
    transcript = result["text"].strip()
    save_result("Wav2Vec2_result", runtime, transcript)
    return transcript, runtime
# ----------------------------------------------------------------------------------------------------------------------
# 5)ESPnet:
def run_espnet(audio_path):
    from espnet_model_zoo.downloader import ModelDownloader
    from espnet2.bin.asr_inference import Speech2Text
    downloader = ModelDownloader()
    tag = "espnet/simpleoier_librispeech_asr_train_asr_conformer7_wavlm_large_raw_en_bpe5000_sp"
    info = downloader.download_and_unpack(tag)
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    model = Speech2Text(asr_train_config=info["asr_train_config"],
                        asr_model_file=info["asr_model_file"],
                        lm_train_config=info["lm_train_config"],
                        lm_file=info["lm_file"],
                        device=device)
    speech, rate = sf.read(audio_path)
    start = time.time()
    result = model(speech)
    runtime = time.time() - start
    transcript = result[0][0].strip()
    save_result("ESPnet_result", runtime, transcript)
    return transcript, runtime
# ----------------------------------------------------------------------------------------------------------------------
# 6)DeepSpeech:
def run_deepspeech(audio_path):
    import deepspeech
    model = deepspeech.Model(str("deepspeech-0.9.3-models.pbmm"))
    model.enableExternalScorer(str("deepspeech-0.9.3-models.scorer"))
    audio, rate = sf.read(audio_path)
    audio = (audio * 32767).astype(np.int16)
    start = time.time()
    transcript = model.stt(audio)
    runtime = time.time() - start
    save_result("DeepSpeech_result", runtime, transcript)
    return transcript, runtime
# ----------------------------------------------------------------------------------------------------------------------
# 7)Main:
if OUTPUT_FILE.exists():
    OUTPUT_FILE.unlink()
processed_audio = preprocess_audio(AUDIO_FILE, PROCESSED_AUDIO)
run_whisper(processed_audio)
run_wav2vec2(processed_audio)
run_espnet(processed_audio)
run_deepspeech(processed_audio)
