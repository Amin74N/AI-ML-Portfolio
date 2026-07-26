import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
# Loading audio file with Librosa:
scale_file = "audio/scale.wav"
scale, sr = librosa.load(scale_file, sr= None)
print("Signal Shape:", scale.shape)
# ----------------------------------------------------------------------------------------------------------------------
# Calculating Zero Crossing Rate (ZCR):
zero_crossings = librosa.zero_crossings(scale, pad= False)
print("All Zero Crossings:", sum(zero_crossings))
# ----------------------------------------------------------------------------------------------------------------------
# Visualizing Zero Crossing Rate (ZCR):
n0, n1 = 15000, 16000
sample_zero_crossings = librosa.zero_crossings(scale[n0:n1], pad= False)
print("Sample Zero Crossings:", sum(sample_zero_crossings))
plt.figure(figsize = (10,5))
plt.plot(scale[n0:n1])
plt.grid(True)
plt.title("Sample Zero Crossings")
plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# Calculating Spectral Rolloff:
spectral_rolloff = librosa.feature.spectral_rolloff(y= scale+0.01, sr= sr)[0]
print("Spectral Rolloff:", spectral_rolloff.shape)
# ----------------------------------------------------------------------------------------------------------------------
# Visualizing Spectral Rolloff:
frames = np.arange(len(spectral_rolloff))
t = librosa.frames_to_time(frames, sr= sr, hop_length= 512)
plt.figure(figsize = (10,5))
librosa.display.waveshow(scale, sr= sr, alpha= 0.3)
def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))
plt.plot(t, normalize(spectral_rolloff), color= 'r')
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("Spectral Rolloff")
plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# Calculating Chroma:
chromagram = librosa.feature.chroma_stft(y= scale, sr= sr, hop_length= 512)
print("Chroma Frequencies:", chromagram.shape)
# ----------------------------------------------------------------------------------------------------------------------
# Visualizing Chroma:
plt.figure(figsize = (10,5))
librosa.display.specshow(chromagram, sr= sr, x_axis= "time", y_axis= "chroma", hop_length= 512, cmap= "coolwarm")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("Chroma Frequencies")
plt.show()
