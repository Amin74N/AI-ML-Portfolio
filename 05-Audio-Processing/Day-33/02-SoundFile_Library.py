import soundfile as sf
# ----------------------------------------------------------------------------------------------------------------------
# Reading a WAV audio file:
input_audio, sr = sf.read("audio/input.wav")
print("Shape:", input_audio.shape)
print("Sample Rate:", sr)
# ----------------------------------------------------------------------------------------------------------------------
# Saving a copy of the WAV audio file:
sf.write("audio/soundfile_output.wav", input_audio, sr)
# ----------------------------------------------------------------------------------------------------------------------
# Changing the format of the WAV audio file:
sf.write("audio/soundfile_output.mp3", input_audio, sr)
