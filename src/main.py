from pipeline_3d import Pipeline3D
from camera_queue import CameraQueue
from projectile_sim import projectile_motion_with_air_resistance, v0, launch_angle, initial_height, k, mass

# Testing stuff
from threading import Thread




def test_camera(queue):
    time_of_flight, times, x_values, y_values = \
    projectile_motion_with_air_resistance(v0, launch_angle,initial_height, k, mass)
    with open("../out/sim", "w") as f:
        for i in range(len(x_values)):
            f.write(f"{x_values[i]} {y_values[i]} {times[i]}\n")
            
    for i in range(len(x_values)):
        queue.put_frame((x_values[i], y_values[i], 0, times[i]))

# from camera import Camera
if __name__ == "__main__":

    # Test code with dummy data
    queue = CameraQueue()

    Thread(target=test_camera, args=(queue,)).start()
    
    # oak_d = Camera(queue)
    pipeline = Pipeline3D(queue)

    # camera.run()
    pipeline.run()