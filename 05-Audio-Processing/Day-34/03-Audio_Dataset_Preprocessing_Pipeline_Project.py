import librosa, librosa.display
import soundfile as sf
from pathlib import Path
import pandas as pd
import numpy as np
# ----------------------------------------------------------------------------------------------------------------------
audio_info = []
output_folder = Path("project/outputs")
output_folder.mkdir(parents= True, exist_ok= True)
# ----------------------------------------------------------------------------------------------------------------------
# 1) Dataset Loading:
input_folder = Path("project/inputs")
files = sorted([f for f in input_folder.iterdir() if f.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg", ".m4a")])
for file in files:
    try:
        input_audio, orig_sr = librosa.load(file, sr= None, mono= False)
        orig_duration = librosa.get_duration(y= input_audio, sr= orig_sr)
        channels = 1 if input_audio.ndim == 1 else input_audio.shape[0]
        num_samples = input_audio.shape[-1]
# ----------------------------------------------------------------------------------------------------------------------
# 2) Dataset Preprocessing:
        mono_input_audio = librosa.to_mono(input_audio) if channels > 1 else input_audio
        target_sr = 16000
        resampled_audio = librosa.resample(mono_input_audio, orig_sr= orig_sr, target_sr= target_sr)
        trimmed, _ = librosa.effects.trim(resampled_audio, top_db= 20)
        max_amp = np.max(np.abs(trimmed))
        normalized = 0.99 * trimmed / max_amp if max_amp > 0 else trimmed
# ----------------------------------------------------------------------------------------------------------------------
# 3) Audio Exporting:
        output_file = output_folder / f"{file.stem}_processed.wav"
        sf.write(output_file, normalized, target_sr)
        resampled_duration = librosa.get_duration(y= normalized, sr= target_sr)
        output_num_samples = len(normalized)
# ----------------------------------------------------------------------------------------------------------------------
# 4) Report Generating:
        audio_info.append({"Input File": file.name,
                           "Output File": output_file.name,
                           "Input Sample Rate (Hz)": orig_sr,
                           "Output Sample Rate (Hz)": target_sr,
                           "Input Samples": num_samples,
                           "Output Samples": output_num_samples,
                           "Input Duration (s)": orig_duration,
                           "Output Duration (s)": resampled_duration,
                           "Input Channels": channels,
                           "Output Channels": 1})
    except Exception as e:
        print(f"Error processing {file.name}: {e}")
df = pd.DataFrame(audio_info)
print(df)
df.to_csv("project/outputs/processed_audio_information.csv", index= False)
