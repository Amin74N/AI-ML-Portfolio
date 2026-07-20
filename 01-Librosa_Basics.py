import wave
# ----------------------------------------------------------------------------------------------------------------------
# Reading an existing WAV file and display its audio properties:
obj = wave.open("audio/1MG_Example.wav", "rb")                                # Opening a WAV file in read-binary mode
print("Number of Channels:", obj.getnchannels())                              # Number of audio channels (1 = Mono, 2 = Stereo)
print("Sample Width:", obj.getsampwidth())                                    # Sample size in bytes
print("Frame Rate:", obj.getframerate())                                      # Sampling rate (samples per second)
print("Number of Frames:", obj.getnframes())                                  # Total number of audio samples (frames)
print("Parameters:", obj.getparams())                                         # Displaying all WAV file parameters
print("Duration:", obj.getnframes() / obj.getframerate())                     # Calculating audio duration in seconds
frames = obj.readframes(-1)                                                   # Reading all audio frames as raw bytes
print("Frames Type:", type(frames))                                           # Data-type of the audio buffer
print("Element Type:", type(frames[0]))                                       # Data-type of a single byte
bytes_per_frame = obj.getnchannels() * obj.getsampwidth()                     # Number of bytes contained in one frame
print("Number of Frames:", len(frames) / bytes_per_frame)                     # Calculating the total number of frames
obj.close()                                                                   # Closing the file
# ----------------------------------------------------------------------------------------------------------------------
# Creating a new WAV file using the audio frames from the original file:
obj_new = wave.open("audio/New_1MG_Example.wav", "wb")                        # Creating a new WAV file in write-binary mode
obj_new.setnchannels(1)                                                       # Output will be Mono
obj_new.setsampwidth(2)                                                       # 16-bit (2 bytes) audio
obj_new.setframerate(16000.0)                                                 # Sampling rate = 16 kHz
obj_new.writeframes(frames)                                                   # Writing the copied audio frames to the new file
print("New File's Duration:", obj_new.getnframes() / obj_new.getframerate())  # Calculating audio duration in seconds
obj_new.close()                                                               # Closing the file
