from gtts import gTTS
import os
import time

class AudioGreeter():
    def __init__(self, save_path):
        self.save_path = save_path
        self.previous_time = -30

    def greet(self, name):
        if not self.check_time():
            return

        tts = gTTS("Hello {}".format(name))
        filename = "{}/greet.mp3".format(self.save_path)

        tts.save(filename)
        os.system("mpg321 {}".format(filename))

    def check_time(self):
        current_time = time.time()
        if current_time - self.previous_time < 10:
            return False
        else:
            self.previous_time = current_time
            return True
