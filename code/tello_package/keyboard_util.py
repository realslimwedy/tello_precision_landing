import time


def keyboard_rc(me, rc_values, pygame_instance, speed):
    py_g = pygame_instance
    lr, fb, ud, yv = rc_values
    speed = speed

    if py_g.get_key('e') and not me.is_flying:
        me.takeoff()
        return 0, 0, 0, 0
    elif py_g.get_key('q') and me.is_flying:
        me.land()
        time.sleep(2)
        return 0, 0, 0, 0

    if py_g.get_key('RIGHT'):
        lr = speed
    elif py_g.get_key('LEFT'):
        lr = -speed
    if py_g.get_key('UP'):
        fb = speed
    elif py_g.get_key('DOWN'):
        fb = -speed
    if py_g.get_key('w'):
        ud = speed
    elif py_g.get_key('s'):
        ud = -speed
    if py_g.get_key('d'):
        yv = speed
    elif py_g.get_key('a'):
        yv = -speed

    return lr, fb, ud, yv


def save_image_key_pressed(pygame_instance):
    py_g = pygame_instance
    if py_g.get_key('SPACE'):
        print('SPACE pressed - Image saving...')
        return True
    return False


def video_capture_key_pressed(video_capture_on, pygame_instance):
    py_g = pygame_instance
    if py_g.get_key('v'):
        print('v pressed: Video capture...')
        return True
    return False


def exit_app_key_pressed(me, pygame_instance):
    py_g = pygame_instance
    if py_g.get_key('ESCAPE'):
        return True
    return False


def autopilot_key_pressed(me, pygame_instance):
    py_g = pygame_instance
    if py_g.get_key('p'):
        print("p pressed: AUTOPILOT...")
        return True
    return False


def takeoff_phase_key_pressed(me, pygame_instance):
    py_g = pygame_instance
    if py_g.get_key('1'):
        return True
    return False

def hover_phase_key_pressed(me, pygame_instance):
    py_g = pygame_instance
    if py_g.get_key('2'):
        return True
    return False

def approach_phase_key_pressed(me, pygame_instance):
    py_g = pygame_instance
    if py_g.get_key('3'):
        return True
    return False


def landing_phase_key_pressed(me, pygame_instance):
    py_g = pygame_instance
    if py_g.get_key('4'):
        return True
    return False


def switch_speed_key_pressed(pygame_instance):
    py_g = pygame_instance
    if py_g.get_key('o'):
        return True
    return False
