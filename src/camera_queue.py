from threading import Lock

class CameraQueue:
    def __init__(self):
        self.frames = []
        self.mutex = Lock()

    def get_frame(self):
        self.mutex.acquire()
        frame = self.frames.pop(0)
        self.mutex.release()
        return frame


    def put_frame(self, frame):
        if len(frame != 4):
            print("Frame needs to be tuple of (x, y, z, timestamp)")
            return False
        self.mutex.acquire()
        self.frames.append(frame)
        self.mutex.release()
        return True

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