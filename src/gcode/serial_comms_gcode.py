import serial
import time
import random
import math

# Taken from https://projecthub.arduino.cc/ansh2919/serial-communication-between-python-and-arduino-663756
# arduino = serial.Serial(port='COM7', baudrate= 115200, timeout=.1)

PAUSE_1S = "G4 P1"
GOTO_ZERO = "G1 X0 Y0"

def clamp(value, lower, upper):
    return max(lower, min(value, upper))

# send XZ coordinate to go to
def gcode_goto(s, x: float, z: float):
    # transform world space (XZ) to grbl/robot space (XY)
    # need to invert one axis
    grbl_x = z
    grbl_y = -x

    # clamp coordinates to [-10, 10] lower and upper limits
    grbl_x = clamp(grbl_x, -10.0, 10.0)
    grbl_y = clamp(grbl_y, -10.0, 10.0)

    # x is a value along the "y-axis", z is a value along the "x axis"
    gcode = 'G1 X' + str(grbl_x) + ' Y' + str(grbl_y) + '\n'
    s.write(bytes(gcode + '\n', 'utf-8'))
    grbl_out = s.readline() # Wait for grbl response with carriage return
    print(' : ' + str(grbl_out.strip()))

# send raw gcode command
def gcode_send(s, command: str):
    s.write(bytes(command + '\n', 'utf-8'))
    grbl_out = s.readline() # Wait for grbl response with carriage return
    print(' : ' + str(grbl_out.strip()))

# initialize grbl connection after opening serial comms
def grbl_init(s):
    # Wake up grbl
    print("Waking up grbl")
    s.write(bytes("\r\n\r\n", 'utf-8'))
    time.sleep(2)   # Wait for grbl to initialize 
    s.flushInput()  # Flush startup text in serial input


    print("Zeroing grbl, setting feed rate")
    gcode_send(s, '$X') # exit lockout state. initially in lockout when limit switches enabled
    gcode_send(s, 'G21') # specify millimeters (kinda... coordinates supplied in cm)
    gcode_send(s, 'G90') # absolute coordinates
    gcode_send(s, 'G17') # XY Plane
    gcode_send(s, 'G94') # units per minute feed rate mode
    gcode_send(s, '$H')  # do homing
    
    gcode_send(s, "F2000") # Set feed rate. for some reason, has to be reduced right after homing
    gcode_send(s, GOTO_ZERO) # Go to zero zero
    gcode_send(s, PAUSE_1S)
    gcode_send(s, "F3750") # Set feed rate for normal operation
    return

# used to verify positional accuracy and translation speed
def random_radius(s):
    max_radius = 10 # 10cm
    r = random.random() * max_radius # [0, 10]
    angle = random.random() * 2*math.pi # [0, 2pi]

    x = r * math.cos(angle)
    z = r * math.sin(angle)

    x = round(x, 2) # 2 decimal points... 0.1mm precision
    z = round(z, 2) # 2 decimal points... 0.1mm precision
    print(x, z)
    gcode_goto(s, x, z)

# used to demonstrate queued movements simulating incrementally better prediction coordinates
# grbl has a 128 character RX buffer that supports this
def random_move(s):
    x = (random.random() - 0.5) * 2 # [-1, 1]
    x *= 7 # [-7, 7]
    z = (random.random() - 0.5) * 2 # [-1, 1]
    z *= 7 # [-7, 7]

    for i in range(5): # roughly simulate increasingly better coordinates
        print(x, z)
        gcode_goto(s, x, z)
        dx = (random.random() - 0.5) * 2 * 2
        dz = (random.random() - 0.5) * 2 * 2
        x += dx
        z += dz

        # ensure safe bounds
        x = clamp(x, -10.0, 10.0)
        z = clamp(z, -10.0, 10.0)


def main():
    # Open grbl serial port
    s = serial.Serial('COM7',115200)

    # initialize grbl connection
    grbl_init(s)

    print("Now accepting user input. Enter q to exit")
    print("Shortcut to send coordinates: c X Y")
    print("Otherwise, send raw GCODE commands")
    while True: # user input loop
        cmd = input("\nEnter command to send to grbl: ").strip()
        print("CMD = " + cmd)
        if cmd == "q": break
        elif cmd == "r": random_move(s)
        elif cmd == "rr": random_radius(s)
        elif cmd == "z": gcode_goto(s, 0, 0)
        elif cmd[0].lower() == "c": # goto X Y
            c, x, z = cmd.split(" ")
            gcode_goto(s, float(x), float(z))
        else: # send RAW gcode command
            gcode_send(s, cmd)
        # gcode_goto(s, 0, 0)
        gcode_send(s, PAUSE_1S)
    

    # Close file and serial port
    s.close()

if __name__ == '__main__':
    main()