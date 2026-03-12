# Disini untuk kodingan FPS
import time

class FPSCounter:
    def __init__(self):
        self.prev_time = time.time()
        self.fps = 0

    def update(self):
        curren_time = time.time()
        delta = curren_time - self.prev_time

        if delta > 0:
            self.fps = 1.0 / delta

        self.prev_time = curren_time
        return self.fps