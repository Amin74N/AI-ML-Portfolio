import wave
import numpy as np
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
# Reading the WAV file and extracting its audio data:
obj = wave.open("audio/5MG_Example.wav", "rb")                    # Opening the WAV file in read-binary mode
sample_freq = obj.getframerate()                                  # Getting the sampling rate (samples per second)
n_samples = obj.getnframes()                                      # Getting the total number of audio frames
n_channels = obj.getnchannels()                                   # Getting the number of audio channels (1 = Mono, 2 = Stereo)
signal_wave = obj.readframes(-1)                                  # Reading all audio frames as raw bytes
obj.close()                                                       # Closing the WAV file
# ----------------------------------------------------------------------------------------------------------------------
# Preparing the audio data for visualization:
duration = n_samples / sample_freq                                # Calculating the audio duration in seconds
signal_array = np.frombuffer(signal_wave, dtype= np.int16)        # Converting raw bytes into a NumPy array
signal_array = signal_array.reshape(-1, n_channels)               # Separating the audio samples into individual channels
if n_channels == 2:                                               # Extract the left and right channels if the audio is stereo
    left_channel = signal_array[:, 0]                             # Extract the left audio channel
    right_channel = signal_array[:, 1]                            # Extract the right audio channel
times = np.linspace(0, duration, num= n_samples, endpoint= False) # Creating the time axis for the waveform
# ----------------------------------------------------------------------------------------------------------------------
# Plotting the waveform of the left audio channel:
plt.figure(figsize= (10,5))                                       # Creating a figure with a custom size
plt.subplot(2, 1, 1)                                              # Create the first subplot
plt.plot(times, left_channel)                                     # Plotting the left-channel waveform
plt.title("Left Channel Waveform of the Audio Signal")            # Setting the plot title
plt.ylabel("Signal Wave")                                         # Labeling the y-axis
plt.xlim(0, duration)                                             # Displaying the entire duration of the audio
# ----------------------------------------------------------------------------------------------------------------------
# Plotting the waveform of the right audio channel:
plt.subplot(2, 1, 2)                                              # Create the second subplot
plt.plot(times, right_channel)                                    # Plotting the right-channel waveform
plt.title("Right Channel Waveform of the Audio Signal")           # Setting the plot title
plt.ylabel("Signal Wave")                                         # Labeling the y-axis
plt.xlabel("Time (seconds)")                                      # Labeling the x-axis
plt.xlim(0, duration)                                             # Displaying the entire duration of the audio
plt.tight_layout()                                                # Adjust spacing between the subplots
plt.show()                                                        # Displaying the waveform plots
