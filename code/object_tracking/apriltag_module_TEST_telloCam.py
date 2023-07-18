import cv2
from apriltag_module import apriltag_center_area
from djitellopy import tello
import time

def main():

    me = tello.Tello()
    me.connect()
    print(f'Battery Level: {me.get_battery()} %')
    time.sleep(0.5)
    me.streamon()

    width, height = 640, 480


    while True:
        img = me.get_frame_read().frame
        img = cv2.resize(img, (width,height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.flip(img, 0)
        
        # Call the main function from apriltag_detection.py
        img, center_x, center_y, area = apriltag_center_area(img)
        
        # Display the frame
        cv2.imshow('Tello Cam', img)
        cv2.waitKey(1)
        
        key = cv2.waitKey(1)
        if key == 27:  # Press Esc to exit
            break

        # Print the tag ID, center coordinates, and area
        print("Center X:", center_x)
        print("Center Y:", center_y)
        print("Area:", area)

    cv2.destroyAllWindows()    
    me.streamoff()

if __name__ == "__main__":
    main()

