from djitellopy import tello
import cv2, math, time

me = tello.Tello()
me.connect()
print(f'Battery Level: {me.get_battery()} %')
time.sleep(0.5)
me.streamon()

cv2.namedWindow("drone")
img = me.get_frame_read()

# Dictionary to store the movement commands for each direction
movement_commands = {
    'left': [0, 0, 0, -100],
    'up': [0, 0, 100, 0],
    'right': [0, 0, 0, 100],
    'down': [0, 0, -100, 0],
    'forward': [0, 100, 0, 0],
    'backward': [0, -100, 0, 0],
    'rotate_left': [-100, 0, 0, 0],
    'rotate_right': [100, 0, 0, 0],
}

# Variable to store the current movement command
current_movement = [0, 0, 0, 0]

while True:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow("drone", img)

    key = cv2.waitKey(1) & 0xff

    # Handle arrow keys
    if key == 81:  # LEFT arrow key
        current_movement = movement_commands['left']
    elif key == 82:  # UP arrow key
        current_movement = movement_commands['up']
    elif key == 83:  # RIGHT arrow key
        current_movement = movement_commands['right']
    elif key == 84:  # DOWN arrow key
        current_movement = movement_commands['down']
    elif key == ord('e'):
        me.takeoff()
    elif key == ord('q'):
        me.land()
        frame_read.stop()
        me.streamoff()
        exit(0)
    elif key == ord('w'):
        current_movement = movement_commands['forward']
    elif key == ord('s'):
        current_movement = movement_commands['backward']
    elif key == ord('a'):
        current_movement = movement_commands['rotate_left']
    elif key == ord('d'):
        current_movement = movement_commands['rotate_right']
    else:
        # If no key is pressed, stop the drone's movement
        current_movement = [0, 0, 0, 0]

    # Send the current movement command to the drone
    me.send_rc_control(current_movement[0], current_movement[1], current_movement[2], current_movement[3])
