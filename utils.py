import os
import time
import playsound
import speech_recognition as sr
from gtts import gTTS


def speak(text):
    try:
        tts = gTTS(text=text, lang="en")
        filename = "voice.mp3"
        tts.save(filename)
        playsound.playsound(filename)
        os.remove(filename)
    except Exception as e:
        print(f'A ERROR {e} has occurred with the speak function.')


def get_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
        said = ""

        try:
            said = r.recognize_google(audio)
            print(f"Input: {said}")
        except Exception as e:
            print(f"ERROR: {e}")

    return said
