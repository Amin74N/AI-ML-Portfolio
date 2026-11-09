import librosa, librosa.display
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
# Loading audio files with Librosa:
scale_file = "audio/scale.wav"
scale, sr = librosa.load(scale_file)
# ----------------------------------------------------------------------------------------------------------------------
# Mel filter banks:
filter_banks = librosa.filters.mel(n_fft= 2048, sr= 22050, n_mels= 10)
print(filter_banks.shape)
plt.figure(figsize= (10,5))
librosa.display.specshow(filter_banks, sr= sr, x_axis= 'linear')
plt.colorbar()
plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# Extracting Mel Spectrogram:
mel_Spectrogram = librosa.feature.melspectrogram(y= scale, sr= sr, n_fft= 2048, hop_length= 512, n_mels= 10)
print(mel_Spectrogram.shape)
log_mel_Spectrogram = librosa.power_to_db(mel_Spectrogram)
plt.figure(figsize= (10,5))
librosa.display.specshow(log_mel_Spectrogram, sr= sr, x_axis= 'time', y_axis= 'mel')
plt.colorbar(format='%+2.f')
plt.show()
