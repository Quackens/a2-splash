import serial
import time
import random
import math

# Taken from https://projecthub.arduino.cc/ansh2919/serial-communication-between-python-and-arduino-663756
# arduino = serial.Serial(port='COM7', baudrate= 115200, timeout=.1)

PAUSE_1S = "G4 P1"
GOTO_ZERO = "G1 X0 Y0"

# send XY coordinate to go to
def gcode_goto(s, x: float, y: float):
    # bounds check
    if not (-10 <= x <= 10): return -1
    elif not (-10 <= y <= 10): return -1

    gcode = 'G1 X' + str(x) + ' Y' + str(y) + '\n'
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

def random_radius(s):
    max_radius = 10 # 10cm
    r = random.random() * max_radius # [0, 10]
    angle = random.random() * 2*math.pi # [0, 2pi]

    x = r * math.cos(angle)
    y = r * math.sin(angle)

    x = round(x, 2) # 2 decimal points... 0.1mm precision
    y = round(y, 2) # 2 decimal points... 0.1mm precision
    print(x, y)
    gcode_goto(s, x, y)

def random_move(s):
    x = (random.random() - 0.5) * 2 # [-1, 1]
    x *= 7 # [-7, 7]
    y = (random.random() - 0.5) * 2 # [-1, 1]
    y *= 7 # [-7, 7]

    for i in range(5): # roughly simulate increasingly better coordinates
        print(x, y)
        gcode_goto(s, x, y)
        dx = (random.random() - 0.5) * 2 * 2
        dy = (random.random() - 0.5) * 2 * 2
        x += dx
        y += dy

        # ensure safe bounds
        if x < -10: x = -10
        elif x > 10: x = 10
        if y < -10: y = -10
        elif y > 10: y = 10 


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
        if cmd == "q": break
        elif cmd == "r": random_move(s)
        elif cmd == "rr": random_radius(s)
        elif cmd[0].lower() == "c": # goto X Y
            c, x, y = cmd.split(" ")
            gcode_goto(s, float(x), float(y))
        else: # send RAW gcode command
            gcode_send(s, cmd)
        # gcode_goto(s, 0, 0)
        gcode_send(s, PAUSE_1S)
    

    # Close file and serial port
    s.close()

if __name__ == '__main__':
    main()