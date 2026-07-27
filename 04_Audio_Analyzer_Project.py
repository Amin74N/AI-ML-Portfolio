import librosa
import wave
from pydub import AudioSegment
import soundfile as sf
# ----------------------------------------------------------------------------------------------------------------------
# 1) Loading an audio file and displaying its basic information (Librosa Library):
print("============ SPEECH ============")
speech_signal, s_sr = librosa.load("audio/speech.wav", sr= None)
print("Sampling Rate     :", s_sr, "Hz")
print(f"Duration          : {librosa.get_duration(y= speech_signal, sr= s_sr):.2f} Seconds")
# ----------------------------------------------------------------------------------------------------------------------
# 2) Reading another audio file and extracting its metadata (Wave Library):
print("============ MUSIC =============")
with wave.open("audio/music.wav", "rb") as music:
    print("Number of Channels:", music.getnchannels())
    print("Frame Rate        :", music.getframerate())
    print("Sample Width      :", music.getsampwidth())
    print("Number of Frames  :", music.getnframes())
    print(f"Duration          : {music.getnframes() / music.getframerate():.2f} Seconds")
# ----------------------------------------------------------------------------------------------------------------------
# 3) Editing audio files and converting the edited file (PyDub Library):
speech_audio = AudioSegment.from_wav("audio/speech.wav")
music_audio = AudioSegment.from_wav("audio/music.wav")
louder_speech = speech_audio + 6
quieter_music = music_audio - 6
trimmed_quieter_music = quieter_music[2000:6000]
fade_out_trimmed_quieter_music = trimmed_quieter_music.fade_out(2000)
fade_in_louder_speech = louder_speech.fade_in(2000)
merged_audio = fade_out_trimmed_quieter_music + fade_in_louder_speech
repeated_merged_audio = merged_audio * 2
repeated_merged_audio.export("audio/converted_output.mp3", format="mp3")
# ----------------------------------------------------------------------------------------------------------------------
# Reading the edited audio and saving the final output (SoundFile Library):
input_audio, sr = sf.read("audio/converted_output.mp3")
sf.write("audio/project_output.wav", input_audio, sr)
print("============= Done =============")
