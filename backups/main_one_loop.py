import pygame, cv2, time
from djitellopy import tello
import tello_package as tp
import object_tracking as ot

if __name__ == '__main__':
    # Your existing code here...

    width, height = 640, 480
    res = (width, height)
    ud_approach_center_area = 20000  # this needs to be calibrated, different for every resolution(?)
    PID = [0.4, 0, 0.4]
    prev_error = (0, 0, 0, 0)
    tp.init_keyboard_control()
    me = tello.Tello()
    me.connect()
    print(f'Battery Level: {me.get_battery()} %')
    time.sleep(0.25)
    me.streamon()
    img = me.get_frame_read().frame
    flight_phase = None  # takeoff, cruise, approach!, land!
    auto_pilot = True  # manual vs. auto pilot
    # maybe implement a data mode to specify image video saving and video feed on/off
    video_feed_on = False
    video_capture_on = False
    exit_program = False
    drone_is_on = True
    rc_params = (0, 0, 0, 0)

    pygame.init()
    screen = pygame.display.set_mode(res)
    pygame.display.set_caption('Video Feed')

    while True:
        # Video feed processing
        img = me.get_frame_read().frame
        img = cv2.resize(img, res)
        img = cv2.flip(img, 0)
        img = ot.apriltag_center_area(img)

        # Convert the image to a format that pygame can display
        img_pygame = pygame.image.frombuffer(img.tobytes(), res, 'RGB')

        # Update the display with the new image
        screen.blit(img_pygame, (0, 0))
        pygame.display.update()

        # Drone control
        rc_params = (0, 0, 0, 0)
        rc_params = tp.keyboard_rc(me, rc_params)
        me.send_rc_control(rc_params[0], rc_params[1], rc_params[2], rc_params[3])

        # Your other main thread tasks here...
        if tp.exit_app_key_pressed(me):
            if me.is_flying == True:
                me.land()
            me.streamoff()
            me.end()
            pygame.quit()
            exit_program = True
            break
