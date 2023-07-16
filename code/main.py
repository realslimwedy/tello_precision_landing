
import pygame, cv2
from djitellopy import tello
import tello_package as tp
import time
import sys

# Connect to TELLO drone wifi
tp.connect_to_wifi("TELLO-9C7357")

# Initialize drone & connect
tp.init_keyboard_control()
me = tello.Tello()
me.connect()
print(f'Battery Level: {me.get_battery()} %')
time.sleep(0.5)

me.streamon()

# Control drone via keyboard
while True:
    
    # Control drone via keyboard
    vals = tp.keyboard_control_drone(me, (0, 0, 0, 0))
    me.send_rc_control(vals[0], vals[1], vals[2], vals[3])
    time.sleep(0.05)

    # Display video feed
    img = me.get_frame_read().frame
    img = cv2.resize(img, (360, 240))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow("Image", img)

    # Save image upon pressing z
    
    tp.save_image(img)
    
    # Quit program upon pressing q
    if tp.exit_app(me):
        break
    

tp.connect_to_wifi("Leev_Marie")
sys.exit()

# next todos:
# - add image/video capture upon button press