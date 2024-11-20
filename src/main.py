from pipeline_3d import Pipeline3D
from camera_queue import CameraQueue
# from camera import Camera
if __name__ == "__main__":
    queue = CameraQueue()

    # oak_d = Camera(queue)
    pipeline = Pipeline3D(queue)

    # camera.run()
    pipeline.run()