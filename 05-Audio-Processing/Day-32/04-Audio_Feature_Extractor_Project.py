import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
# Loading audio files with Librosa:
speech, s_sr = librosa.load("audio/speech.wav", sr= None)
music, m_sr = librosa.load("audio/music.wav", sr= None)
print(f"SPEECH Sampling Rate: {s_sr} Hz")
print(f"MUSIC Sampling Rate: {m_sr} Hz")
print(f"SPEECH Number of Samples: {speech.shape[0]}")
print(f"MUSIC Number of Samples: {music.shape[0]}")
print(f"SPEECH Duration: {speech.shape[0] / s_sr:.2f} Seconds")
print(f"MUSIC Duration: {music.shape[0] / m_sr:.2f} Seconds")
# ----------------------------------------------------------------------------------------------------------------------
# Calculating Zero Crossing Rate (ZCR):
s_zero_crossings = librosa.zero_crossings(speech, pad=False)
m_zero_crossings = librosa.zero_crossings(music, pad=False)
print(f"SPEECH Mean ZCR: {np.mean(s_zero_crossings):.2f}")
print(f"MUSIC Mean ZCR:: {np.mean(m_zero_crossings):.2f}")
# ----------------------------------------------------------------------------------------------------------------------
# Visualizing Zero Crossing Rate (ZCR):
n0, n1 = 15000, 16000
s_sample_zero_crossings = librosa.zero_crossings(speech[n0:n1], pad=False)
m_sample_zero_crossings = librosa.zero_crossings(music[n0:n1], pad=False)
print("SPEECH Sample Zero Crossings:", sum(s_sample_zero_crossings))
print("MUSIC Sample Zero Crossings:", sum(m_sample_zero_crossings))
plt.figure(figsize=(13, 5))
plt.subplot(1, 2, 1)
plt.plot(speech[n0:n1])
plt.grid(True)
plt.title("SPEECH Sample Zero Crossings")
plt.subplot(1, 2, 2)
plt.plot(music[n0:n1])
plt.grid(True)
plt.title("MUSIC Sample Zero Crossings")
plt.tight_layout()
plt.savefig("ZCR.png")
plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# Calculating Spectral Centroid:
FRAME_SIZE = 1024
HOP_LENGTH = 512
sc_speech = librosa.feature.spectral_centroid(y= speech, sr= s_sr, n_fft= FRAME_SIZE, hop_length= HOP_LENGTH)[0]
sc_music = librosa.feature.spectral_centroid(y= music, sr= m_sr, n_fft= FRAME_SIZE, hop_length= HOP_LENGTH)[0]
print(f"SPEECH Mean Spectral Centroid: {np.mean(sc_speech):.2f} Hz")
print(f"MUSIC Mean Spectral Centroid: {np.mean(sc_music):.2f} Hz")
# ----------------------------------------------------------------------------------------------------------------------
# Visualizing Spectral Centroid:
s_frames = np.arange(len(sc_speech))
s_t = librosa.frames_to_time(s_frames, sr= s_sr, hop_length= HOP_LENGTH)
m_frames = np.arange(len(sc_music))
m_t = librosa.frames_to_time(m_frames, sr= m_sr, hop_length= HOP_LENGTH)
plt.figure(figsize = (13,5))
plt.subplot(1, 2, 1)
plt.plot(s_t, sc_speech)
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("SPEECH Spectral Centroid")
plt.subplot(1, 2, 2)
plt.plot(m_t, sc_music)
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("MUSIC Spectral Centroid")
plt.tight_layout()
plt.savefig("Spectral_Centroid.png")
plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# Calculating Spectral Bandwidth:
sb_speech = librosa.feature.spectral_bandwidth(y= speech, sr= s_sr, n_fft= FRAME_SIZE, hop_length= HOP_LENGTH)[0]
sb_music = librosa.feature.spectral_bandwidth(y= music, sr= m_sr, n_fft= FRAME_SIZE, hop_length= HOP_LENGTH)[0]
print(f"SPEECH Mean Spectral Bandwidth: {np.mean(sb_speech):.2f} Hz")
print(f"MUSIC Mean Spectral Bandwidth: {np.mean(sb_music):.2f} Hz")
# ----------------------------------------------------------------------------------------------------------------------
# Visualizing Spectral Bandwidth:
plt.figure(figsize = (13,5))
plt.subplot(1, 2, 1)
plt.plot(s_t, sb_speech)
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("SPEECH Spectral Bandwidth")
plt.subplot(1, 2, 2)
plt.plot(m_t, sb_music)
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("MUSIC Spectral Bandwidth")
plt.tight_layout()
plt.savefig("Spectral_Bandwidth.png")
plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# Calculating Spectral Rolloff:
sr_speech = librosa.feature.spectral_rolloff(y= speech+0.01, sr= s_sr)[0]
sr_music = librosa.feature.spectral_rolloff(y= music+0.01, sr= m_sr)[0]
print(f"SPEECH Mean Spectral Rolloff: {np.mean(sr_speech):.2f} Hz")
print(f"MUSIC Mean Spectral Rolloff: {np.mean(sr_music):.2f} Hz")
# ----------------------------------------------------------------------------------------------------------------------
# Visualizing Spectral Rolloff:
s_frames = np.arange(len(sr_speech))
s_t = librosa.frames_to_time(s_frames, sr= s_sr, hop_length= HOP_LENGTH)
m_frames = np.arange(len(sr_music))
m_t = librosa.frames_to_time(m_frames, sr= m_sr, hop_length= HOP_LENGTH)
plt.figure(figsize = (13,5))
def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))
plt.subplot(1, 2, 1)
librosa.display.waveshow(speech, sr= s_sr, alpha= 0.3)
plt.plot(s_t, normalize(sr_speech), color= 'r')
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("SPEECH Spectral Rolloff")
plt.subplot(1, 2, 2)
librosa.display.waveshow(music, sr= m_sr, alpha= 0.3)
plt.plot(m_t, normalize(sr_music), color= 'r')
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("MUSIC Spectral Rolloff")
plt.tight_layout()
plt.savefig("Spectral_Rolloff.png")
plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# Extracting MFCCs:
s_mfccs = librosa.feature.mfcc(y= speech, n_mfcc= 13, sr= s_sr)
m_mfccs = librosa.feature.mfcc(y= music, n_mfcc= 13, sr= m_sr)
print("SPEECH MFCCs Shape:", s_mfccs.shape)
print("MUSIC MFCCs Shape:", m_mfccs.shape)
# ----------------------------------------------------------------------------------------------------------------------
# Visualizing MFCCs:
plt.figure(figsize = (13,5))
plt.subplot(1, 2, 1)
librosa.display.specshow(s_mfccs, x_axis= 'time', sr= s_sr)
plt.title('SPEECH MFCCs')
plt.subplot(1, 2, 2)
librosa.display.specshow(m_mfccs, x_axis= 'time', sr= m_sr)
plt.title('MUSIC MFCCs')
plt.colorbar(format='%+2f')
plt.tight_layout()
plt.savefig("MFCCs.png")
plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# Calculating Chroma:
s_chromagram = librosa.feature.chroma_stft(y= speech, sr= s_sr, hop_length= HOP_LENGTH)
m_chromagram = librosa.feature.chroma_stft(y= music, sr= m_sr, hop_length= HOP_LENGTH)
print("SPEECH Chroma Shape:", s_chromagram.shape)
print("MUSIC Chroma Shape:", m_chromagram.shape)
# ----------------------------------------------------------------------------------------------------------------------
# Visualizing Chroma:
plt.figure(figsize = (13,5))
plt.subplot(1, 2, 1)
librosa.display.specshow(s_chromagram, sr= s_sr, x_axis= "time", y_axis= "chroma", hop_length= HOP_LENGTH, cmap= "coolwarm")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("SPEECH Chroma Frequencies")
plt.subplot(1, 2, 2)
librosa.display.specshow(m_chromagram, sr= m_sr, x_axis= "time", y_axis= "chroma", hop_length= HOP_LENGTH, cmap= "coolwarm")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("MUSIC Chroma Frequencies")
plt.tight_layout()
plt.savefig("Chroma.png")
plt.show()
