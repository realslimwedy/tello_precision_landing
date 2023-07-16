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

# Translate key input to drone movement
def keyboard_rc(me, rc_values):
    lr, fb, ud, yv = rc_values
    speed = 50 

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
    elif get_key('LEFT'):
        lr = -speed
    if get_key('UP'):
        fb = speed
    elif get_key('DOWN'):
        fb = -speed
    if get_key('w'):
        ud = speed
    elif get_key('s'):
        ud = -speed
    if get_key('d'):
        yv = speed
    elif get_key('a'):
        yv = -speed
    
    return (lr, fb, ud, yv)

# Save image via keyboard
def save_image(img):
    if get_key('SPACE'):
        print('spacebar pressed - saving image...')
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'data/images_saved_by_drone/{time.time()}.jpg', img)
        time.sleep(0.2)

def exit_app(me):
    if get_key('ESCAPE'):
        print("ESC pressed: Landing drone and ending program...")
        return True
    return False
