import pygame, cv2, time, sys, asyncio
from djitellopy import tello
import tello_package as tp
import object_tracking as ot
import argparse


if __name__ == '__main__': 
    # tp.connect_to_wifi("TELLO-9C7357")
    width, height = 640, 480
    resolution = (width, height)
    ud_approach_center_area = 20000 # this needs to be calibrated, different for every resolution(?)
    PID= [0.4, 0, 0.4] 
    prv_error = (0, 0, 0, 0)
    tp.init_keyboard_control()
    me = tello.Tello()
    me.connect()
    print(f'Battery Level: {me.get_battery()} %')
    time.sleep(0.5)
    me.streamon()
    flight_phase=None #takeoff, cruise, approach!, land!
    auto_pilot=True # manual vs. auto pilot
    # maybe implement a data mode to specify image video saving and video feed on/off
    video_feed_on = False
    video_capture_on = False
    exit_program = False
    drone_is_on = True


    while True:
        rc_params= (0, 0, 0, 0)
                        
        # ESC shuts down everything
        if tp.exit_app_key_pressed(me):
            if me.is_flying==True: 
                me.land()
            me.streamoff()
            me.end()
            cv2.destroyAllWindows()
            pygame.quit()
            exit_program = True
            break
    
        if exit_program==True:
            sys.exit()
            
        if tp.video_feed_key_pressed(video_feed_on):
            if not video_feed_on:
                video_feed_on = True
            elif video_feed_on:
                video_feed_on = False

        if video_feed_on:
            img = me.get_frame_read().frame
            img = cv2.resize(img, (width,height))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.flip(img, 0)
            img, _, _ = ot.apriltag_center_area(img)                    
            cv2.imshow('Video Feed', img)
            cv2.waitKey(1)
            #print(f'Center: {center}')
            #print(f'Area: {area}')
            
        elif not video_feed_on:
            cv2.destroyAllWindows()