import pygame
import time
import cv2
import sys

# Initialize pygame
def init_keyboard_control():
    pygame.init()
    win = pygame.display.set_mode((400,400))

# Get key input from user
def get_key(key_name):
    ans = False
    for event in pygame.event.get(): 
        pass
    key_input = pygame.key.get_pressed()
    my_key = getattr(pygame, f'K_{key_name}')
    if key_input[my_key]:
        ans = True
    pygame.display.update()
    return ans

################################

# DIRECT RC CONTROL IN MANUAL FLIGHT MODE

def keyboard_rc(me, rc_values):
    lr, fb, ud, yv = rc_values
    speed = 100 

    # Start and land drone via keyboard
    if get_key('e') and not me.is_flying:
        me.takeoff()
        return (0, 0, 0, 0)
    elif get_key('q') and me.is_flying:
        me.land()
        time.sleep(2)
        return (0, 0, 0, 0)
    
    # Control drone via keyboard
    if get_key('RIGHT'):
        lr = speed
        print('RIGHT pressed...')
    elif get_key('LEFT'):
        lr = -speed
        print('LEFT pressed...')
    if get_key('UP'):
        fb = speed
        print('UP pressed...')
    elif get_key('DOWN'):
        fb = -speed
        print('DOWN pressed...')
    if get_key('w'):
        ud = speed
        print('w pressed...')
    elif get_key('s'):
        ud = -speed
        print('s pressed...')
    if get_key('d'):
        yv = speed
        print('d pressed...')
    elif get_key('a'):
        yv = -speed
        print('a ...')
    
    return (lr, fb, ud, yv)

################################

# VIDEO FEED & CAPTURE

# Save image via keyboard
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
    return False

################################

# EXIT APP

def exit_app_key_pressed(me):
    if get_key('ESCAPE'):
        print("ESC pressed: Landing drone and ending program...")
        print(f'Remaining Battery Level: {me.get_battery()} %')
        return True
    return False

################################

# FLIGHT PHASES

def approach_phase_key_pressed():
    if get_key('5'):
        print("5 pressed: Approach phase activated...")
        print(f'Remaining Battery Level: {me.get_battery()} %')
        return True
    return False

def safe_landing_key_pressed():
    pass