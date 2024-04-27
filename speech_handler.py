import time
import winsound
from os import path
import speech_recognition as sr

class speechHandler:
    # class to handle speech recognition by voice or file
    def __init__(self, verbose = True):
        self.verbose = verbose
        self.r = sr.Recognizer()
        self.r.energy_threshold = 500
        
    def recognizeSpeechFromMic(self, trigger = None, return_audio = False):
        try:
            with sr.Microphone() as source:
                if trigger != None:
                    try:
                        winsound.Beep(trigger[0], trigger[1]) # freq and duration
                    except:
                        pass
                audio = self.r.listen(source, timeout = None)

            text = self.r.recognize_google(audio)
            print("Heard: " + str(text))
        except:
            text = None
            audio = None
         
        print("sending: " + str(text))
        
        if return_audio:
            return str(text), audio

        return str(text)
            
