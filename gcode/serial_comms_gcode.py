import serial
import time

# Taken from https://projecthub.arduino.cc/ansh2919/serial-communication-between-python-and-arduino-663756
# arduino = serial.Serial(port='COM7', baudrate= 115200, timeout=.1)

def write_read(x): 
    arduino.write(bytes(x, 'utf-8')) 
    time.sleep(0.05) 
    data = arduino.readline() 
    return data

def gcode_goto(x: float, y: float):
    # bounds checking
    if not (-10 <= x <= 10): return -1
    elif not (-10 <= y <= 10): return -1

    gcode = 'G1 X' + str(x) + ' Y' + str(y) + '\n'
    write_read(gcode)
    return

def gcode_send(s, command: str):
    s.write(bytes(command + '\n', 'utf-8'))
    grbl_out = s.readline() # Wait for grbl response with carriage return
    print(' : ' + str(grbl_out.strip()))


PAUSE_1S = "G4 P1"
GOTO_ZERO = "G1 X0 Y0"
def main():
    # Open grbl serial port
    s = serial.Serial('COM7',115200)

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
    
    gcode_send(s, "F2000") # Set feed rate. for some reason, has to be low right after homing
    gcode_send(s, GOTO_ZERO) # Go to zero zero
    gcode_send(s, PAUSE_1S)

    gcode_send(s, "F3750") # Set feed rate for normal operation
    print("Now accepting user input. Enter q to exit")
    while True: # user input loop
        cmd = input("\nEnter command to send to grbl: ")
        if cmd == "q": break

        gcode_send(s, cmd)
        gcode_send(s, PAUSE_1S)
        gcode_send(s, GOTO_ZERO)

    # input("  Press <Enter> to exit and disable grbl.") 
    # Close file and serial port
    s.close()

if __name__ == '__main__':
    main()