import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
# Loading audio files with Librosa:
scale_file = "audio/scale.wav"
debussy_file = "audio/debussy.wav"
redhot_file = "audio/redhot.wav"
duke_file = "audio/duke.wav"
scale, sr = librosa.load(scale_file)
debussy, _ = librosa.load(debussy_file)
redhot, _ = librosa.load(redhot_file)
duke, _ = librosa.load(duke_file)
# ----------------------------------------------------------------------------------------------------------------------
# Extracting Short-Time Fourier Transform (STFT):
FRAME_SIZE = 2048
HOP_SIZE = 512
S_scale = librosa.stft(y= scale, n_fft= FRAME_SIZE, hop_length= HOP_SIZE)
print(S_scale.shape)
print(type(S_scale[0][0]))
# ----------------------------------------------------------------------------------------------------------------------
# Calculating the spectrogram:
Y_scale = np.abs(S_scale) ** 2
print(Y_scale.shape)
print(type(Y_scale[0][0]))
# ----------------------------------------------------------------------------------------------------------------------
# Visualizing the spectrogram:
def plot_spectrogram (Y, sr, hop_length, y_axis= "linear"):
    plt.figure(figsize= (10,5))
    librosa.display.specshow(Y, sr= sr, hop_length= hop_length, x_axis= "time", y_axis= y_axis)
    plt.colorbar(format= '%+2.f')
plot_spectrogram(Y_scale, sr, HOP_SIZE)
plt.title("Linear Spectrogram (Scale)")
plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# Log-Amplitude spectrogram:
Y_log_scale = librosa.power_to_db(Y_scale)
plot_spectrogram(Y_log_scale, sr, HOP_SIZE)
plt.title("Log-Amplitude Spectrogram (Scale)")
plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# Log-Frequency spectrogram:
plot_spectrogram(Y_log_scale, sr, HOP_SIZE, y_axis= "log")
plt.title("Log-Frequency Spectrogram (Scale)")
plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# Visualizing songs from different genres:
S_debussy = librosa.stft(y= debussy, n_fft= FRAME_SIZE, hop_length= HOP_SIZE)
Y_debussy = np.abs(S_debussy) ** 2
Y_log_debussy = librosa.power_to_db(Y_debussy)
plot_spectrogram(Y_log_debussy, sr, HOP_SIZE, y_axis= "log")
plt.title("Log-Frequency Spectrogram (Debussy)")
plt.show()
S_redhot = librosa.stft(y= redhot, n_fft= FRAME_SIZE, hop_length= HOP_SIZE)
Y_redhot = np.abs(S_redhot) ** 2
Y_log_redhot = librosa.power_to_db(Y_redhot)
plot_spectrogram(Y_log_redhot, sr, HOP_SIZE, y_axis= "log")
plt.title("Log-Frequency Spectrogram (Red Hot Chili Peppers)")
plt.show()
S_duke = librosa.stft(y= duke, n_fft= FRAME_SIZE, hop_length= HOP_SIZE)
Y_duke = np.abs(S_duke) ** 2
Y_log_duke = librosa.power_to_db(Y_duke)
plot_spectrogram(Y_log_duke, sr, HOP_SIZE, y_axis= "log")
plt.title("Log-Frequency Spectrogram (Duke Ellington)")
plt.show()
