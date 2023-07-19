import pygame, cv2, time, sys, asyncio, copy
from djitellopy import tello
import tello_package as tp
import object_tracking as ot
from apriltag_module import apriltag_center_area
from pupil_apriltags import Detector
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
    img = me.get_frame_read().frame
    flight_phase=None #takeoff, cruise, approach!, land!
    auto_pilot=True # manual vs. auto pilot
    # maybe implement a data mode to specify image video saving and video feed on/off
    video_feed_on = False
    video_capture_on = False
    exit_program = False
    drone_is_on = True
    rc_params= (0, 0, 0, 0)


    async def main():
        
        async def video_stream():
            global img
            while True:
                img = me.get_frame_read().frame
                await asyncio.sleep(1./30)

        async def show_video_feed():
            global img
            while True:
                if img is not None:
                    img = cv2.resize(img, (width,height))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.flip(img, 0)
                    img = ot.apriltag_center_area(img)
                    cv2.imshow('Video Feed', img)
                    cv2.waitKey(1)
                    await asyncio.sleep(0.05)

        async def drone_control():
            while True:
                rc_params= tp.keyboard_rc(me, (0, 0, 0, 0))
                me.send_rc_control(rc_params[0], rc_params[1], rc_params[2], rc_params[3])
                
                if tp.exit_app_key_pressed(me):
                    if me.is_flying==True: 
                        me.land()
                    me.streamoff()
                    me.end()
                    cv2.destroyAllWindows()
                    pygame.quit()
                    exit_program = True
                    break

                await asyncio.sleep(0.01)
            
            if exit_program==True:
                sys.exit()

    
        await asyncio.gather(video_stream(),  show_video_feed(), drone_control())

    asyncio.run(main())

