import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
# Loading audio file with Librosa:
audio_file = "audio/debussy.wav"
signal, sr = librosa.load(audio_file)
print("Signal Shape:", signal.shape)
# ----------------------------------------------------------------------------------------------------------------------
# Extracting MFCCs:
mfccs = librosa.feature.mfcc(y= signal, n_mfcc= 13, sr= sr)
print("MFCCs Shape:", mfccs.shape)
# ----------------------------------------------------------------------------------------------------------------------
# Visualizing MFCCs:
plt.figure(figsize = (10,5))
librosa.display.specshow(mfccs, x_axis= 'time', sr= sr)
plt.colorbar(format='%+2f')
plt.title('MFCCs')
plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# Calculating Delta & Delta2 MFCCs:
delta_mfccs = librosa.feature.delta(mfccs)
delta2_mfccs = librosa.feature.delta(mfccs, order= 2)
print("Delta MFCCs Shape:", delta_mfccs.shape)
print("Delta Delta MFCCs Shape:", delta2_mfccs.shape)
# ----------------------------------------------------------------------------------------------------------------------
# Visualizing Delta MFCCs & Delta Delta MFCCs:
plt.figure(figsize = (10,5))
librosa.display.specshow(delta_mfccs, x_axis= 'time', sr= sr)
plt.colorbar(format='%+2f')
plt.title('Delta MFCCs')
plt.show()
plt.figure(figsize = (10,5))
librosa.display.specshow(delta2_mfccs, x_axis= 'time', sr= sr)
plt.colorbar(format='%+2f')
plt.title('Delta Delta MFCCs')
plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# Calculating Comprehensive MFCCs:
comprehensive_mfccs = np.concatenate((mfccs, delta_mfccs, delta2_mfccs))
print("Comprehensive MFCCs Shape:", comprehensive_mfccs.shape)
