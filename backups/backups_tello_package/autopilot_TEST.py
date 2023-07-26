import cv2
import autopilot as ap
from djitellopy import tello
import time

import sys
import os

# Get the parent folder path
parent_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../code'))

# Add the parent folder to the system path
sys.path.append(parent_folder_path)

# Import the scripts from the parent folder
from object_tracking import apriltag_module as ot


# Use the imported scripts in your script


def main():

    me = tello.Tello()
    me.connect()
    print(f'Battery Level: {me.get_battery()} %')
    time.sleep(0.5)
    me.streamon()

    width, height = 640, 480
    res = (width, height)
    ud_approach_center_area = 20000
    rc_params = (0, 0, 0, 0)
    PID= [0.4, 0, 0.4] 
    prv_error = (0, 0, 0, 0)

    while True:
        img = me.get_frame_read().frame
        img = cv2.resize(img, (width,height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.flip(img, 0)
        
        # Call the main function from apriltag_detection.py
        img, center, area = ot.apriltag_center_area(img)
        
        # Display the frame
        cv2.imshow('Tello Cam', img)
        cv2.waitKey(1)

        # Test the autopilot
        error = ap.get_alignment_error(center, area, ud_approach_center_area, res)
        rc_params = ap.track_apriltag(rc_params, center, area, PID, error, prv_error)
        prv_error = error
        
        # print data
        print(f'rc_params: {rc_params}')

        key = cv2.waitKey(1)
        if key == 27:  # Press Esc to exit
            break
        

    cv2.destroyAllWindows()    
    me.streamoff()

if __name__ == "__main__":
    main()

