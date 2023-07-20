import pygame, cv2, time, sys, asyncio
from djitellopy import tello
import tello_package as tp
import object_tracking as ot

if __name__ == '__main__':

    width, height = 640, 480
    res = (width, height)
    exit_program = False

    # autopilot
    flight_phase = None  # takeoff, cruise, approach!, land!
    auto_pilot = False  # manual vs. auto pilot
    ud_approach_center_area = width * height * 0.2 ** 2  # this needs to be calibrated, different for every resolution(?)
    PID = [0.4, 0, 0.4]
    prv_error = (0, 0, 0, 0)

    # drone control
    tp.init_keyboard_control()
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


    async def main():

        async def show_video_feed():
            global img
            while True:
                img = me.get_frame_read().frame
                img = cv2.resize(img, res)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.flip(img, 0)

                await asyncio.sleep(1. / 30)

                cv2.imshow('Video Feed', img)
                cv2.waitKey(1)

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
