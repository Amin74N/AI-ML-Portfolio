import librosa, librosa.display
import soundfile as sf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
# 1) Loading an audio file:
pd.options.display.float_format = "{:.2f}".format
input_audio_info = []
output_audio_info = []
input_audio, orig_sr = librosa.load("audio/input.wav", sr= None, mono= True)
target_sr = 16000
resampled_audio = librosa.resample(input_audio, orig_sr= orig_sr, target_sr= target_sr)
orig_duration = librosa.get_duration(y= input_audio, sr= orig_sr)
num_samples = len(input_audio)
input_audio_info.append({"File-Name": "input.wav",
                         "Sample-Rate": orig_sr,
                         "Number-of-Samples": num_samples,
                         "Duration": orig_duration,})
input_df = pd.DataFrame(input_audio_info)
print("Original Audio Information:\n", input_df.T.rename(columns={0: ""}))

# ----------------------------------------------------------------------------------------------------------------------
# 2) Silence-Removal of the audio file:
trimmed, _ = librosa.effects.trim(resampled_audio, top_db= 20)
# ----------------------------------------------------------------------------------------------------------------------
# 3) Normalizing the audio file:
normalized = trimmed / np.max(np.abs(trimmed))
# ----------------------------------------------------------------------------------------------------------------------
# 4) Saving output audio files:
sf.write("audio/output.wav", normalized, target_sr)
resampled_duration = librosa.get_duration(y= normalized, sr= target_sr)
output_num_samples = len(normalized)
output_audio_info.append({"File-Name": "output.wav",
                          "Sample-Rate": target_sr,
                          "Number-of-Samples": output_num_samples,
                          "Duration": resampled_duration,})
output_df = pd.DataFrame(output_audio_info)
print("-----------------------------\nProcessed Audio Information:\n", output_df.T.rename(columns={0: ""}))
# ----------------------------------------------------------------------------------------------------------------------
# 5) Visualizing audio files waveforms:
plt.figure(figsize= (13,10))
plt.subplot(2,1,1)
librosa.display.waveshow(input_audio, sr= orig_sr)
plt.title("Original Input")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.subplot(2,1,2)
librosa.display.waveshow(normalized, sr= target_sr)
plt.title("Processed Output")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()
