import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
# Loading audio file with Librosa:
audio, sr = librosa.load("audio/debussy.wav", sr= None)
# ----------------------------------------------------------------------------------------------------------------------
# Waveform:
librosa.display.waveshow(audio, sr= sr)
plt.title("Waveform")
plt.savefig("IMG1_Waveform.png")
plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# FFT:
fft = np.fft.rfft(audio)
magnitude = np.abs(fft)
freqs = np.fft.rfftfreq(len(audio), d= 1/sr)
plt.plot(freqs, magnitude)
plt.title("FFT")
plt.savefig("IMG2_FFT.png")
plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# Spectrum:
S = librosa.stft(audio, n_fft = 2048, hop_length = 512)
first_frame = np.abs(S[:,0])
freqs = librosa.fft_frequencies(sr= sr, n_fft= 2048)
plt.plot(freqs, first_frame)
plt.title("Spectrum")
plt.savefig("IMG3_Spectrum.png")
plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# Spectrogram:
SG = librosa.stft(audio)
SG = np.abs(SG) ** 2
librosa.display.specshow(SG, sr= sr, x_axis= "time", y_axis= "linear")
plt.colorbar()
plt.title("Spectrogram")
plt.savefig("IMG4_Spectrogram.png")
plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# Log Spectrogram:
LSG = librosa.stft(audio)
LSG_power = np.abs(LSG) ** 2
LSG_db = librosa.power_to_db(LSG_power, ref= np.max)
librosa.display.specshow(LSG_db, sr= sr, x_axis= "time", y_axis= "log")
plt.colorbar()
plt.title("Log Spectrogram")
plt.savefig("IMG5_Log_Spectrogram")
plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# Mel Spectrogram:
MSG = librosa.feature.melspectrogram(y= audio, sr= sr, n_fft= 2048, hop_length= 512)
librosa.display.specshow(MSG, sr= sr, x_axis= "time", y_axis= "mel")
plt.colorbar()
plt.title("Mel Spectrogram")
plt.savefig("IMG6_Mel_Spectrogram.png")
plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# Log Mel Spectrogram:
MSG = librosa.feature.melspectrogram(y= audio, sr= sr, n_fft= 2048, hop_length= 512)
MSG_db = librosa.power_to_db(MSG, ref= np.max)
librosa.display.specshow(MSG_db, sr= sr, x_axis= "time", y_axis= "mel")
plt.colorbar()
plt.title("Log Mel Spectrogram")
plt.savefig("IMG7_Log_Mel_Spectrogram.png")
plt.show()
