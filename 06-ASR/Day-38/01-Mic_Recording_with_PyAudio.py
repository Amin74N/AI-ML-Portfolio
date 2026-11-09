import pyaudio
import wave
# ----------------------------------------------------------------------------------------------------------------------
FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5
# ----------------------------------------------------------------------------------------------------------------------
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=FRAMES_PER_BUFFER)
print("Listening...")
frames = []
for i in range(0, int(RATE / FRAMES_PER_BUFFER * RECORD_SECONDS)):
    data = stream.read(FRAMES_PER_BUFFER)
    frames.append(data)
stream.stop_stream()
stream.close()
p.terminate()
# ----------------------------------------------------------------------------------------------------------------------
audio = wave.open('test1.wav', 'wb')
audio.setnchannels(CHANNELS)
audio.setsampwidth(p.get_sample_size(FORMAT))
audio.setframerate(RATE)
audio.writeframes(b"".join(frames))
audio.close()
