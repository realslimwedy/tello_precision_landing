# this is the keyboard_util.py file

import time
import tkinter as tk

def on_key_press(window, event):
    key = event.keysym
    if key == 'Escape':
        print("ESC key pressed: Exiting the application...")
        window.destroy()
    elif key == 'e':
        print("e: Taking off...")
        # Perform takeoff action here
        #return (0, 0, 0, 0)
    elif key == 'q':
        print("q key pressed: Landing...")
        # Perform landing action here
        #time.sleep(2)
        #return (0, 0, 0, 0)
    elif key == 'Right':
        print("RIGHT")
        # Perform right movement action here
        #return (speed, 0, 0, 0)
    elif key == 'Left':
        print("LEFT")
        # Perform left movement action here
        #return (-speed, 0, 0, 0)
    elif key == 'Up':
        print("UP")
        # Perform forward movement action here
        #return (0, speed, 0, 0)
    elif key == 'Down':
        print("DOWN")
        # Perform backward movement action here
        #return (0, -speed, 0, 0)
    elif key == 'w':
        print("E")
        # Perform upward movement action here
        #return (0, 0, speed, 0)
    elif key == 's':
        print("S")
        # Perform downward movement action here
        #return (0, 0, -speed, 0)
    elif key == 'd':
        print("D")
        # Perform right rotation action here
        #return (0, 0, 0, speed)
    elif key == 'a':
        print("a")
        # Perform left rotation action here
        #return (0, 0, 0, -speed)
    else:
        print(f"KEY NOT ASSIGNED: {key}")
        # return (0, 0, 0, 0)

def init_keyboard_control():
    window = tk.Tk()
    window.bind("<KeyPress>", lambda event: on_key_press(window, event))
    window.mainloop() # within an async



def main():
    init_keyboard_control()

if __name__ == '__main__':
    main()


#################################


'''# Function to handle key press events
def keyboard_rc(me, rc_values, speed):
    lr, fb, ud, yv = rc_values
    # Get the key code from the event
    key_code = event.keycode
    # how do i get the event?
    #

    # Perform actions based on the key code
    
    else:
        return (0, 0, 0, 0)'''




################################

# VIDEO FEED & CAPTURE

'''# Save image via keyboard
def save_image_key_pressed():
    if get_key('SPACE'):
        print('spacebar pressed - saving image...')
        return True
    return False

# Start and stop video feed via keyboard
def video_feed_key_pressed(video_feed_on):
    if get_key('f'):
        if video_feed_on:
            print('f pressed: Video feed stopped...')
        elif not video_feed_on:
            print('f pressed: Video feed started...')
        return True
    return False

# Start and stop video capture via keyboard
def video_capture_key_pressed(video_capture_on):
    if get_key('v'):
        if video_capture_on:
            print('v pressed: Video capture stopped...')
        elif not video_capture_on:
            print('v pressed: Video capture started...')
        return True
    return False'''

################################

# EXIT APP

'''def exit_app_key_pressed(me):
    if get_key('ESCAPE'):
        print("ESC pressed: Landing drone and ending program...")
        print(f'Remaining Battery Level: {me.get_battery()} %')
        return True
    return False'''





