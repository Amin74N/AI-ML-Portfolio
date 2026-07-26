import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
# Loading audio files with Librosa:
debussy_file = "audio/debussy.wav"
redhot_file = "audio/redhot.wav"
duke_file = "audio/duke.wav"
debussy, sr = librosa.load(debussy_file)
redhot, _ = librosa.load(redhot_file)
duke, _ = librosa.load(duke_file)
# ----------------------------------------------------------------------------------------------------------------------
# Calculating Spectral Centroid:
FRAME_SIZE = 1024
HOP_LENGTH = 512
sc_debussy = librosa.feature.spectral_centroid(y= debussy, sr= sr, n_fft= FRAME_SIZE, hop_length= HOP_LENGTH)[0]
sc_redhot = librosa.feature.spectral_centroid(y= redhot, sr= sr, n_fft= FRAME_SIZE, hop_length= HOP_LENGTH)[0]
sc_duke = librosa.feature.spectral_centroid(y= duke, sr= sr, n_fft= FRAME_SIZE, hop_length= HOP_LENGTH)[0]
print("Spectral Centroid (Debussy):", sc_debussy.shape)
print("Spectral Centroid (RedHot):", sc_redhot.shape)
print("Spectral Centroid (Duke):", sc_duke.shape)
# ----------------------------------------------------------------------------------------------------------------------
# Visualizing Spectral Centroid:
frames = np.arange(len(sc_debussy))
t = librosa.frames_to_time(frames, sr= sr, hop_length= HOP_LENGTH)
plt.figure(figsize = (10,5))
plt.plot(t, sc_debussy, color= 'b')
plt.plot(t, sc_redhot, color= 'r')
plt.plot(t, sc_duke, color= 'y')
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("Spectral Centroid")
plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# Calculating Spectral Bandwidth:
sb_debussy = librosa.feature.spectral_bandwidth(y= debussy, sr= sr, n_fft= FRAME_SIZE, hop_length= HOP_LENGTH)[0]
sb_redhot = librosa.feature.spectral_bandwidth(y= redhot, sr= sr, n_fft= FRAME_SIZE, hop_length= HOP_LENGTH)[0]
sb_duke = librosa.feature.spectral_bandwidth(y= duke, sr= sr, n_fft= FRAME_SIZE, hop_length= HOP_LENGTH)[0]
print("Spectral Bandwidth (Debussy):", sb_debussy.shape)
print("Spectral Bandwidth (RedHot):", sb_redhot.shape)
print("Spectral Bandwidth (Duke):", sb_duke.shape)
# ----------------------------------------------------------------------------------------------------------------------
# Visualizing Spectral Bandwidth:
plt.figure(figsize = (10,5))
plt.plot(t, sb_debussy, color= 'b')
plt.plot(t, sb_redhot, color= 'r')
plt.plot(t, sb_duke, color= 'y')
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("Spectral Bandwidth")
plt.show()
