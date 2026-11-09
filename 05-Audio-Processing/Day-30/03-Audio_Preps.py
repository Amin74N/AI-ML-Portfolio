import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
# ----------------------------------------------------------------------------------------------------------------------
# Loading the audio file:
file = "audio/5MG_Example.wav"                                            # Loading the audio file
signal, sr = librosa.load(file, sr=22050)                                 # resampling the audio file to 22050 Hz
# ----------------------------------------------------------------------------------------------------------------------
# Waveform Visualization (Displaying the amplitude of the audio signal over time):
plt.figure(figsize=(10, 4))                                               # Creating a figure with a custom size
librosa.display.waveshow(signal, sr=sr)                                   # Plotting the waveform of the audio signal
plt.xlabel("Time")                                                        # Labeling the x-axis
plt.ylabel("Amplitude")                                                   # Labeling the y-axis
plt.title("Audio Waveform")                                               # Setting the plot title
plt.show()                                                                # Displaying the waveform plot
# ----------------------------------------------------------------------------------------------------------------------
# FFT -> Frequency Spectrum (Converting the time-domain signal into frequency-domain representation):
fft = np.fft.fft(signal)                                                  # Applying Fast Fourier Transform (FFT)
magnitude = np.abs(fft)                                                   # Calculating the magnitude of each frequency component
frequency = np.linspace(0, sr, len(magnitude))                            # Creating the frequency axis
left_frequency = frequency[:int(len(frequency) / 2)]                      # Extracting positive frequencies
left_magnitude = magnitude[:int(len(frequency) / 2)]                      # Extracting corresponding magnitudes
plt.figure(figsize=(10, 4))                                               # Creating a figure with a custom size
plt.plot(left_frequency, left_magnitude)                                  # Plotting the frequency spectrum
plt.xlabel("Frequency")                                                   # Labeling the x-axis
plt.ylabel("Magnitude")                                                   # Labeling the y-axis
plt.title("Frequency Spectrum")                                           # Setting the plot title
plt.show()                                                                # Displaying the spectrum plot
# ----------------------------------------------------------------------------------------------------------------------
# STFT -> Spectrogram (Representing how frequency components change over time):
n_fft = 2048                                                              # Number of FFT samples per frame
hop_length = 512                                                          # Number of samples between consecutive frames
stft = librosa.core.stft(signal, hop_length= hop_length, n_fft= n_fft)    # Applying Short-Time Fourier Transform
spectrogram = np.abs(stft)                                                # Converting complex values into magnitudes
log_spectrogram = librosa.amplitude_to_db(spectrogram)                    # Converting amplitude values to decibel scale
librosa.display.specshow(log_spectrogram, sr= sr, hop_length= hop_length) # Displaying the spectrogram
plt.xlabel("Time")                                                        # Labeling the x-axis
plt.ylabel("Frequency")                                                   # Labeling the y-axis
plt.title("Spectrogram")                                                  # Setting the plot title
plt.colorbar()                                                            # Adding color scale
plt.show()                                                                # Displaying the spectrogram
# ----------------------------------------------------------------------------------------------------------------------
# MFCCs(Extracting Mel-Frequency Cepstral Coefficients for speech analysis):
MFCCs = librosa.feature.mfcc(y= signal, n_fft= n_fft,                     # Calculating MFCC features
                             hop_length= hop_length, n_mfcc= 13)
librosa.display.specshow(MFCCs, sr= sr, hop_length= hop_length)           # Displaying MFCC features
plt.xlabel("Time")                                                        # Labeling the x-axis
plt.ylabel("MFCC")                                                        # Labeling the y-axis
plt.title("MFCC Features")                                                # Setting the plot title
plt.colorbar()                                                            # Adding color scale
plt.show()                                                                # Displaying the spectrogram
