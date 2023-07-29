import pygame, cv2, sys, asyncio
from ultralytics import YOLO

import tello_package as tp
import object_tracking as ot

if __name__ == '__main__':

    # init main variables
    width, height = 640, 480  # 640, 480 <> 320, 240
    res = (width, height)
    exit_program = False

    # Init YOLO
    classes = ["apple", "banana", "background", "book", "person"]

    model_obj_det = YOLO('../code/vision_package/yoloV8_models/yolov8n.pt')
    print("Object Detection Model loaded")
    model_seg = YOLO('../code/vision_package/yoloV8_models/yolov8n-seg.pt')
    print("Segmentation Model loaded")

    # Init Landing Zone Finder
    lz_finder = ot.LzFinder(model_obj_det=model_obj_det, model_seg=model_seg, res=res, label_list=classes, use_seg=True,
                            r_landing_factor=8, stride=75)
    print(lz_finder)

    # init pygame
    py_game = tp.Pygame(res=res)
    screen = pygame.display.set_mode(res)
    screen_variables_names_units = {
        'names': {
            'battery_level': 'Battery Level',
            'flight_phase': 'Flight Phase',
            'auto_pilot_armed': 'Auto-Pilot Armed',
            'speed': 'Speed',
            'temperature': 'Temperature',
            'flight_time': 'Flight Time'
        },
        'units': {
        'battery_level': '%',
        'flight_phase': '',
        'auto_pilot_armed': '',
        'speed': '',
        'temperature': '°C',
        'flight_time': 'sec'
        }
    }
    print(py_game)

    # init drone
    drone = tp.Drone(res=res, mirror_down=True, speed=50)
    drone.power_up()
    drone_is_one = drone.drone_is_on
    rc_values = (0, 0, 0, 0)
    img = drone.get_frame()
    print(drone)

    # init auto_pilot
    auto_pilot = tp.Autopilot(res=res, speed=25, apriltag_factor=1)
    auto_pilot_armed = auto_pilot.auto_pilot_armed
    prev_error = (0, 0, 0, 0)
    print(auto_pilot)


    # async coroutine
    async def main():

        async def show_video_feed():
            global img, drone_is_one
            while drone_is_one:
                img = drone.get_frame()

                await asyncio.sleep(1. / 30) # await can only be used inside an async function

                # video feed via cv2
                '''img_cv2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cv2.imshow('Video Feed', img_cv2)
                cv2.waitKey(1)'''

                # img_yolo = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # img_yolo, obstacles = objectDetector.infer_image(height, width, img)

                '''yolo_results = model(img_yolo)
                annotated_yolo_frame = yolo_results[0].plot()
                cv2.imshow("YOLOv8 Inference", annotated_yolo_frame)'''

                # video feed via pygame
                battery_level, temperature, flight_time, _, _ = drone.get_drone_sensor_data()

                py_game.display_video_feed(screen, img)
                py_game.display_multiple_status(screen=screen, screen_variables_names_units=screen_variables_names_units,
                                                v_pos=10, h_pos=10,
                                                battery_level=battery_level, flight_phase=drone.flight_phase,
                                                auto_pilot_armed=auto_pilot.auto_pilot_armed, speed=drone.SPEED, temperature=temperature,
                                                flight_time=flight_time)

                pygame.display.flip()

                if tp.exit_app_key_pressed(drone.me, py_game):
                    drone_is_one = False
                    drone.power_down()
                    cv2.destroyAllWindows()
                    pygame.quit()
                    exit_program = True
                    break

            if exit_program == True:
                sys.exit()

        async def drone_control():
            global img, prev_error
            while drone_is_one:
                rc_params = (0, 0, 0, 0)
                target_xy = None
                area = 0

                if tp.auto_pilot_key_pressed(drone.me, py_game):
                    if not auto_pilot.auto_pilot_armed:
                        auto_pilot.arm_autopilot()
                    elif auto_pilot.auto_pilot_armed:
                        auto_pilot.disarm_autopilot()

                if tp.takeoff_phase_key_pressed(drone.me, py_game):
                    drone.flight_phase = "Take-off"
                elif tp.approach_phase_key_pressed(drone.me, py_game):
                    drone.flight_phase = "Approach"
                elif tp.landing_phase_key_pressed(drone.me, py_game):
                    drone.flight_phase = "Landing"

                # if auto_pilot.autopilot_armed:
                if drone.flight_phase == "Take-off":
                    pass
                # drone.me.send_rc_control(0, 0, 0, auto_pilot.speed)
                elif drone.flight_phase == "Approach":
                    img, target_xy, area = ot.apriltag_center_area(img)

                elif drone.flight_phase == "Landing":
                    pass
                    landing_zone_xy, img, risk_map = lz_finder.get_final_lz(img)
                    area = None

                if target_xy is not None:
                    error = auto_pilot.get_alignment_error(target_xy, area)
                    rc_params = auto_pilot.track_target(rc_params, target_xy, error, prev_error, area)
                    prev_error = error

                if tp.switch_speed_key_pressed(py_game):
                    if drone.SPEED == 50:
                        drone.SPEED = 100
                    elif drone.SPEED == 100:
                        drone.SPEED = 50

                rc_params = tp.keyboard_rc(drone.me, rc_params, py_game, drone.SPEED)

                drone.me.send_rc_control(rc_params[0], rc_params[1], rc_params[2], rc_params[3])

                await asyncio.sleep(0.01)

        await asyncio.gather(show_video_feed(), drone_control())

    # create asyncio event loop
    asyncio.run(main())
