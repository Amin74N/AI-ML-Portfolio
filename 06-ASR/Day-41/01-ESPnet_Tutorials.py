from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.s2t_inference import Speech2Text
import torch
import librosa
import soundfile as sf
import numpy as np
# ----------------------------------------------------------------------------------------------------------------------
# 1)Select device:
device = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------------------------------------------------------------------------------------------------
# 2)Initialize ESPnet Model Zoo:
downloader = ModelDownloader()
# ----------------------------------------------------------------------------------------------------------------------
# 3)Select pretrained ASR model:
tag = "espnet/owsm_v4_base_102M"
# ----------------------------------------------------------------------------------------------------------------------
# 4)Download/load model files:
info = downloader.download_and_unpack(tag)
# ----------------------------------------------------------------------------------------------------------------------
# 5)Load pretrained model:
model = Speech2Text(s2t_train_config=info["s2t_train_config"], s2t_model_file=info["s2t_model_file"], device=device,)
# ----------------------------------------------------------------------------------------------------------------------
# 6)Read original audio:
audio_path = "audio/English.wav"
audio, rate = sf.read(audio_path)
print("Original audio:")
print("Sample rate:", rate)
print("Audio shape:", audio.shape)
# ----------------------------------------------------------------------------------------------------------------------
# 7)Audio preprocessing:
audio_mono = np.mean(audio, axis=1) if audio.ndim > 1 else audio
audio_16k = librosa.resample(audio_mono, orig_sr=rate, target_sr=16000) if rate != 16000 else audio_mono
processed_audio_path = "audio/English_16k.wav"
sf.write(processed_audio_path,audio_16k,16000)
print("Audio preprocessing completed!")
# ----------------------------------------------------------------------------------------------------------------------
# 8)Load processed audio:
speech, sample_rate = sf.read(processed_audio_path)
print("Preprocessed audio:")
print("Sample rate:", sample_rate)
print("Samples:", len(speech))
print("Duration:", len(speech) / sample_rate, "seconds")
# ----------------------------------------------------------------------------------------------------------------------
# 9)Run ASR inference:
result = model(speech)
# ----------------------------------------------------------------------------------------------------------------------
# 10)Extract transcription:
text = result[0][0]
clean_text = (text.replace("<eng>", "").replace("<asr>", "").replace("<notimestamps>", "").strip())
# ----------------------------------------------------------------------------------------------------------------------
# 11)Show result:
print("Model:", tag)
print("Model information:", info.keys())
print("Device:", device)
print("Language: English")
print("Task: ASR")
print("Transcription:")
print(clean_text)
