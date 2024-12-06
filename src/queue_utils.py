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
        return self.frames.get()
    
    def put_coord(self, frame):
        self.frames.put(frame)
        return True

    def get_length(self):
        return self.frames.qsize()
    
    def reset_queue(self):
        self.frames = Queue()


# class FrameQueue:
#     def __init__(self):
#         self.frames = []
#         self.mutex = Lock()

#     def get_frame(self):
#         self.mutex.acquire()
#         if len(self.frames) == 0:
#             self.mutex.release()
#             return None
#         frame = self.frames.pop(0)
#         self.mutex.release()
#         return frame


#     def put_frame(self, frame):
#         self.mutex.acquire()
#         self.frames.append(frame)
#         self.mutex.release()
#         return True


class CameraQueue2D:
    def __init__(self):
        self.frames = []
        self.mutex = Lock()

    def get_frame(self):
        self.mutex.acquire()
        if len(self.frames) == 0:
            self.mutex.release()
            return None
        frame = self.frames.pop(0)
        self.mutex.release()
        return frame


    def put_frame(self, frame):
        if len(frame) != 3:
            print("Frame needs to be tuple of (x, y, timestamp)")
            return False
        self.mutex.acquire()
        self.frames.append(frame)
        self.mutex.release()
        return True

class CameraQueue3D:
    def __init__(self):
        self.frames = []
        self.mutex = Lock()

    def get_frame(self):
        self.mutex.acquire()
        if len(self.frames) == 0:
            self.mutex.release()
            return None
        frame = self.frames.pop(0)
        self.mutex.release()
        return frame


    def put_frame(self, frame):
        if len(frame) != 4:
            print("Frame needs to be tuple of (x, y, z, timestamp)")
            return False
        self.mutex.acquire()
        self.frames.append(frame)
        self.mutex.release()
        return True


# class CoordQueue2D:
#     def __init__(self):
#         self.frames = []
#         self.mutex = Lock()

#     def get_coord(self):
#         self.mutex.acquire()
#         if len(self.frames) == 0:
#             self.mutex.release()
#             return None
#         frame = self.frames.pop(0)
#         self.mutex.release()
#         return frame


#     def put_coord(self, frame):
#         if len(frame) != 2:
#             print("Frame needs to be tuple of (x, y)")
#             return False
#         self.mutex.acquire()
#         self.frames.append(frame)
#         self.mutex.release()
#         return True
    # async def get_frame(self):
    #     frame = await self.frames.get()

    #     # Frame has tuple (x, y, z, timestamp)
    #     if len(frame != 4):
    #         print("Frame needs to be tuple of (x, y, z, timestamp)")
    #         return None
        
    #     return await self.frames.get()


    # async def put_frame(self, frame):
        
    #      # Frame has tuple (x, y, z, timestamp)
    #     if len(frame != 4):
    #         print("Frame needs to be tuple of (x, y, z, timestamp)")
    #         return None
        
    #     await self.frames.put(frame)