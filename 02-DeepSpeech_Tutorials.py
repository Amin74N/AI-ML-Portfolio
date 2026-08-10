import deepspeech
import numpy as np
import librosa
import soundfile as sf
# ----------------------------------------------------------------------------------------------------------------------
# 1)Load model & scorer:
model_path = "deepspeech-0.9.3-models.pbmm"
model = deepspeech.Model(model_path)
scorer_path = "deepspeech-0.9.3-models.scorer"
model.enableExternalScorer(scorer_path)
# ----------------------------------------------------------------------------------------------------------------------
# 2)Read audio:
audio_path = "audio/English.wav"
audio, sr = sf.read(audio_path)
print("Original Audio:")
print("Sample rate:", sr)
print("Shape:", audio.shape)
# ----------------------------------------------------------------------------------------------------------------------
# 3)Preprocessing:
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)
if sr != 16000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
audio = (audio * 32767).astype(np.int16)
print("Processed Audio:")
print("Sample rate: 16000")
print("Samples:", len(audio))
# ----------------------------------------------------------------------------------------------------------------------
# 4)ASR:
text = model.stt(audio)
print("\nTranscription:")
print(text)
