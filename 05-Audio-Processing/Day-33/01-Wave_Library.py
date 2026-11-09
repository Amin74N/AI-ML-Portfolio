import wave
import numpy as np
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
# Reading an existing WAV file and extracting metadata:
input_audio = wave.open("audio/input.wav", "rb")
print("INPUT:")
print("Number of Channels:", input_audio.getnchannels())
print("Sample Width:", input_audio.getsampwidth())
print("Frame Rate:", input_audio.getframerate())
print("Number of Frames:", input_audio.getnframes())
print("Parameters:", input_audio.getparams())
print("Duration:", input_audio.getnframes() / input_audio.getframerate())
frames = input_audio.readframes(-1)
print("Frames Type:", type(frames))
print("Element Type:", type(frames[0]))
print("Length of Frames:", len(frames))
bytes_per_frame = input_audio.getnchannels() * input_audio.getsampwidth()
print("Number of Frames:", len(frames) / bytes_per_frame)
input_audio.close()
# ----------------------------------------------------------------------------------------------------------------------
# Creating a new WAV file using the audio frames:
output_audio = wave.open("audio/wave_output.wav", "wb")
output_audio.setnchannels(2)
output_audio.setsampwidth(2)
output_audio.setframerate(22050.0)
output_audio.writeframes(frames)
output_audio.close()
output_audio = wave.open("audio/wave_output.wav", "rb")
print("OUTPUT:")
print("Parameters:", output_audio.getparams())
print("Duration:", output_audio.getnframes() / output_audio.getframerate())
# ----------------------------------------------------------------------------------------------------------------------
# Reading the WAV file and extracting audio data:
sample_freq = output_audio.getframerate()
n_samples = output_audio.getnframes()
n_channels = output_audio.getnchannels()
signal_wave = output_audio.readframes(-1)
output_audio.close()
# ----------------------------------------------------------------------------------------------------------------------
# Preparing the audio data for visualization:
duration = n_samples / sample_freq
signal_array = np.frombuffer(signal_wave, dtype= np.int16)
signal_array = signal_array.reshape(-1, n_channels)
left_channel = signal_array[:, 0]
right_channel = signal_array[:, 1]
times = np.linspace(0, duration, num= n_samples, endpoint= False)
# ----------------------------------------------------------------------------------------------------------------------
# Plotting the waveform of the left audio channel:
plt.figure(figsize= (13,5))
plt.subplot(2, 1, 1)
plt.plot(times, left_channel)
plt.title("Left Channel Waveform of the Output Audio Signal")
plt.ylabel("Signal Wave")
plt.xlabel("Time (seconds)")
plt.xlim(0, duration)
# ----------------------------------------------------------------------------------------------------------------------
# Plotting the waveform of the right audio channel:
plt.subplot(2, 1, 2)
plt.plot(times, right_channel)
plt.title("Right Channel Waveform of the Output Audio Signal")
plt.ylabel("Signal Wave")
plt.xlabel("Time (seconds)")
plt.xlim(0, duration)
plt.tight_layout()
plt.show()
