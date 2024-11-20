import asyncio

class CameraQueue:
    def __init__(self):
        self.frames = asyncio.Queue()

    async def get_frame(self):
        frame = await self.frames.get()

        # Frame has tuple (x, y, z, timestamp)
        if len(frame != 4):
            print("Frame needs to be tuple of (x, y, z, timestamp)")
            return None
        
        return await self.frames.get()


    async def put_frame(self, frame):
        
         # Frame has tuple (x, y, z, timestamp)
        if len(frame != 4):
            print("Frame needs to be tuple of (x, y, z, timestamp)")
            return None
        
        await self.frames.put(frame)