from gtts import gTTS
import os
import time

def play_audio(filename):
    os.system('mpg321 "{}"'.format(filename))

class AudioGreeter():
    def __init__(self, save_path):
        self.save_path = save_path
        self.previous_time = 0

    def greet(self, name):
        filename = "{}/greet {}.mp3".format(self.save_path, name)

        if not self.seconds_passed(10):
            return

        if os.path.isfile(filename):
            play_audio(filename)
            return

        tts = gTTS("Hello {}".format(name))
        tts.save(filename)
        play_audio(filename)

    def seconds_passed(self, seconds):
        current_time = time.time()
        if current_time - self.previous_time < seconds:
            return False
        else:
            self.previous_time = current_time
            return True
