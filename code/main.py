
import pygame
from djitellopy import tello
import tello_package as tp
import time

# Connect to TELLO drone wifi
tp.connect_to_wifi("TELLO-9C7357")
#time.sleep(7)

# Initialize drone & connect
tp.init_keyboard_control()
me = tello.Tello()
me.connect()
print(me.get_battery())
time.sleep(0.5)

#me.streamon()

''' test flight
me.takeoff()
me.send_rc_control(0,50,0,0)
sleep(2)
me.send_rc_control(0,0,0,30)
sleep(2)
me.send_rc_control(0,0,0,0)
me.land()
'''

# control drone via keyboard
while True:
    vals=tp.keyboard_control_drone(me, (0,0,0,0))
    me.send_rc_control(vals[0],vals[1],vals[2],vals[3])
    time.sleep(0.05)