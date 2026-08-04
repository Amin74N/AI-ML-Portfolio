import speech_recognition as sr
from gtts import gTTS
import playsound
# ----------------------------------------------------------------------------------------------------------------------
def get_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)
        said = ""
        try:
            said = r.recognize_google(audio, language="en-US")
            print(said)
        except Exception as e:
            print("Exception: " + str(e))
    return said
# ----------------------------------------------------------------------------------------------------------------------
def speak(text):
    tts = gTTS(text=text, lang='en')
    filename = "test2.mp3"
    tts.save(filename)
    playsound.playsound(filename)
# ----------------------------------------------------------------------------------------------------------------------
text = get_audio()
if "hello" in text.lower():
    speak("Hello, how are you?")
else:
    speak("Sorry, I couldn't hear you.")
