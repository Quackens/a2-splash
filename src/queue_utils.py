from threading import Lock
from queue import Queue

class FrameQueue:
    def __init__(self):
        self.frames = Queue()

    def get_frame(self):
        return self.frames.get()
    
    def put_frame(self, frame):
        self.frames.put(frame)
        return True

    def get_length(self):
        return self.frames.qsize()
    

class CameraQueue2D:
    def __init__(self):
        self.frames = Queue()

    def get_frame(self):
        return self.frames.get()
    
    def put_frame(self, frame):
        self.frames.put(frame)
        return True


class CoordQueue2D:
    def __init__(self):
        self.frames = Queue()

    def get_coord(self):
        try:
            return self.frames.get()
        except:
            return None
        
    def put_coord(self, frame):
        self.frames.put(frame)
        return True

    def get_length(self):
        return self.frames.qsize()
    
    def reset_queue(self):
        self.frames = Queue()

class SignalStart:
    def __init__(self):
        self.start = False
        self.reset = False

    def get_start(self):
        return self.start
    
    def set_start(self, start):
        self.start = start
        return True

    def get_reset(self):
        return self.reset
    
    def set_reset(self, reset):
        self.reset = reset
        return True