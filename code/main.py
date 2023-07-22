import pygame, cv2, time, sys, asyncio
from ultralytics import YOLO
from djitellopy import tello

import tello_package as tp
import object_tracking as ot

def display_status(text):
    font = pygame.font.SysFont(None, 25)
    status_text = font.render(text, True, (255, 255, 255))
    screen.blit(status_text, (10, 10))  # Adjust the position as needed

if __name__ == '__main__':

    model = YOLO('./object_tracking/yolo_models/yolov8n.pt')

    width, height = 640, 480
    res = (width, height)
    exit_program = False

    # autopilot
    flight_phase = None  # takeoff, cruise, approach!, land!
    auto_pilot = False  # manual vs. auto pilot
    ud_approach_center_area = width * height * 0.15 ** 2  # this needs to be calibrated, different for every resolution(?)
    PID = [0.4, 0, 0.4]
    prv_error = (0, 0, 0, 0)

    # drone control
    tp.init_keyboard_control()
    screen = pygame.display.set_mode(res)

    rc_params = (0, 0, 0, 0)

    # initial connection
    # tp.connect_to_wifi("TELLO-9C7357")
    me = tello.Tello()
    me.connect()
    print(f'Battery Level: {me.get_battery()} %')
    time.sleep(0.5)
    me.streamon()
    img = me.get_frame_read().frame
    drone_is_on = True

    # Video capture & data
    # maybe implement a data mode to specify image video saving and video feed on/off
    video_feed_on = False
    video_capture_on = False

    #pygame
    battery_font = pygame.font.SysFont(None, 25)


    async def main():

        async def show_video_feed():
            global img, auto_pilot
            while True:
                img = me.get_frame_read().frame
                img = cv2.resize(img, res)
                img = cv2.flip(img, 0)

                await asyncio.sleep(1. / 30)

                # video feed via cv2
                '''img_cv2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cv2.imshow('Video Feed', img_cv2)
                cv2.waitKey(1)'''

                img_yolo = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                yolo_results = model(img_yolo)
                annotated_yolo_frame = yolo_results[0].plot()
                cv2.imshow("YOLOv8 Inference", annotated_yolo_frame)

                # video feed via pygame
                img_py = cv2.flip(img, 1) # flip horizontally
                img_py = cv2.rotate(img_py, cv2.ROTATE_90_COUNTERCLOCKWISE)

                img_py = pygame.surfarray.make_surface(img_py)
                screen.blit(img_py, (0, 0))
                display_status("Auto Pilot: On" if auto_pilot else "Auto Pilot: Off")
                battery_level = me.get_battery()
                battery_text = battery_font.render(f'Battery Level: {battery_level}%', True, (255, 255, 255))
                screen.blit(battery_text, (10, 40))  # Adjust the position as needed

                pygame.display.flip()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

                if tp.exit_app_key_pressed(me):
                    if me.is_flying == True:
                        me.land()
                    me.streamoff()
                    me.end()
                    cv2.destroyAllWindows()
                    pygame.quit()
                    exit_program = True
                    break

            if exit_program == True:
                sys.exit()

        async def drone_control():
            global img, rc_params, prv_error, auto_pilot
            while True:
                rc_params = (0, 0, 0, 0)

                if tp.approach_phase_key_pressed(me):
                    if not auto_pilot:
                        auto_pilot = True
                        print('Auto pilot activated')
                    elif auto_pilot:
                        auto_pilot = False
                        print('Auto pilot deactivated')

                if auto_pilot:
                    img, center, area = ot.apriltag_center_area(img)

                    if center is not None and area is not None:
                        error = tp.get_alignment_error(center, area, ud_approach_center_area, res)
                        rc_params = tp.track_apriltag(rc_params, center, area, PID, error, prv_error)
                        prv_error = error

                    else:
                        print("Center and/or area is None. Skipping auto-pilot processing.")

                rc_params = tp.keyboard_rc(me, rc_params)

                me.send_rc_control(rc_params[0], rc_params[1], rc_params[2], rc_params[3])

                await asyncio.sleep(0.01)

        await asyncio.gather(show_video_feed(), drone_control())


    asyncio.run(main())
