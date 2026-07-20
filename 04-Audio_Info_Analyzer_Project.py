import librosa, librosa.display
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
# Loading the audio file using Librosa:
audio, sr = librosa.load("audio/5MG_Example.wav", sr= None) # Loading the audio file preserving its original sampling rate
# ----------------------------------------------------------------------------------------------------------------------
# Displaying the sampling rate:
print("Sampling Rate:", sr, "Hz")                           # The sampling rate in Hertz (Hz)
# ----------------------------------------------------------------------------------------------------------------------
# Displaying the total number of audio samples:
print("Number of Samples:",audio.shape[0])                  # The total number of audio samples
# ----------------------------------------------------------------------------------------------------------------------
# Displaying the audio file duration:
print("Duration:",audio.shape[0] / sr, "Seconds")           # Calculating the audio duration in seconds
# ----------------------------------------------------------------------------------------------------------------------
# Displaying the data types of the audio:
print("Audio Type:", type(audio))                           # Displaying the data type of the audio signal
print("Sampling Rate Type:", type(sr))                      # Display the data type of the sampling rate
# ----------------------------------------------------------------------------------------------------------------------
# Waveform Visualization (Displaying the amplitude of the audio signal over time):
plt.figure(figsize=(10, 4))                                 # Creating a figure with a custom size
librosa.display.waveshow(audio, sr=sr)                      # Plotting the waveform of the audio signal
plt.xlabel("Time (seconds)")                                # Labeling the x-axis
plt.ylabel("Amplitude")                                     # Labeling the y-axis
plt.title("Audio Waveform")                                 # Setting the plot title
plt.savefig("Waveform.png")                                 # Saving the waveform plot
plt.show()                                                  # Displaying the waveform plot
