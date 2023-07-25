import time

def keyboard_rc(me, rc_values, pygame_instance):
    py_g=pygame_instance
    lr, fb, ud, yv = rc_values
    speed = 50

    # Start and land drone via keyboard
    if py_g.get_key('e') and not me.is_flying:
        me.takeoff()
        return 0, 0, 0, 0
    elif py_g.get_key('q') and me.is_flying:
        me.land()
        time.sleep(2)
        return 0, 0, 0, 0
    
    # Control drone via keyboard
    if py_g.get_key('RIGHT'):
        lr = speed
        print('RIGHT pressed...')
    elif py_g.get_key('LEFT'):
        lr = -speed
        print('LEFT pressed...')
    if py_g.get_key('UP'):
        fb = speed
        print('UP pressed...')
    elif py_g.get_key('DOWN'):
        fb = -speed
        print('DOWN pressed...')
    if py_g.get_key('w'):
        ud = speed
        print('w pressed...')
    elif py_g.get_key('s'):
        ud = -speed
        print('s pressed...')
    if py_g.get_key('d'):
        yv = speed
        print('d pressed...')
    elif py_g.get_key('a'):
        yv = -speed
        print('a pressed...')
    
    return lr, fb, ud, yv

################################

# VIDEO FEED & CAPTURE

# Save image via keyboard
def save_image_key_pressed(pygame_instance):
    py_g = pygame_instance
    if py_g.get_key('SPACE'):
        print('spacebar pressed - saving image...')
        return True
    return False

# Start and stop video feed via keyboard
def video_feed_key_pressed(video_feed_on, pygame_instance):
    py_g = pygame_instance
    if py_g.get_key('f'):
        if video_feed_on:
            print('f pressed: Video feed stopped...')
        elif not video_feed_on:
            print('f pressed: Video feed started...')
        return True
    return False

# Start and stop video capture via keyboard
def video_capture_key_pressed(video_capture_on, pygame_instance):
    py_g = pygame_instance
    if py_g.get_key('v'):
        if video_capture_on:
            print('v pressed: Video capture stopped...')
        elif not video_capture_on:
            print('v pressed: Video capture started...')
        return True
    return False

################################

# EXIT APP

def exit_app_key_pressed(me, pygame_instance):
    py_g = pygame_instance
    if py_g.get_key('ESCAPE'):
        print("ESC pressed: Landing drone and ending program...")
        print(f'Remaining Battery Level: {me.get_battery()} %')
        return True
    return False

################################

# FLIGHT PHASES

def autopilot_key_pressed(me, pygame_instance):
    py_g=pygame_instance
    if py_g.get_key('p'):
        print("p pressed: AUTOPILOT...")
        return True
    return False

def takeoff_phase_key_pressed(me, pygame_instance):
    py_g=pygame_instance
    if py_g.get_key('1'):
        print("1 pressed: Approach Phase...")
        return True
    return False

def approach_phase_key_pressed(me, pygame_instance):
    py_g=pygame_instance
    if py_g.get_key('3'):
        print("3 pressed: Approach Phase...")
        return True
    return False

def landing_phase_key_pressed(me, pygame_instance):
    py_g=pygame_instance
    if py_g.get_key('4'):
        print("3 pressed: Landing Phase...")
        return True
    return False