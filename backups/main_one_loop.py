import pygame, cv2, sys, time
from ultralytics import YOLO

import tello_package as tp
import object_tracking as ot
from utils import rolling_average

if __name__ == '__main__':

    # init main variables
    WIDTH, HEIGHT = 320, 240  # (1280, 720), (640, 480), (320, 240)
    RES = (WIDTH, HEIGHT)
    exit_program = False
    img_obj_det = None
    LABELS_YOLO = ot.labels_yolo

    # init AprilTag Center Finder
    apriltag_finder = ot.ApriltagFinder(resolution=RES)

    # Init YOLO
    CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'sports ball', 'bottle', 'wine glass', 'fork', 'knife', 'spoon', 'bowl', 'banana',
               'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
               'couch', 'potted plant', 'bed', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'book', 'clock', 'vase', 'scissors', 'toothbrush', 'background']

    MODEL_OBJ_DET = YOLO('../code/object_tracking/yoloV8_models/yolov8n.pt')
    print("Object Detection Model loaded")
    MODEL_SEG = YOLO('../code/object_tracking/yoloV8_models/yolov8n-seg.pt')
    print("Segmentation Model loaded")

    # Init Landing Zone Finder
    lz_finder = ot.LzFinder(model_obj_det=MODEL_OBJ_DET, model_seg=MODEL_SEG, labels=LABELS_YOLO, res=RES,
                            label_list=CLASSES, use_seg=False, r_landing_factor=8, stride=75)
    print(lz_finder)

    # init pygame
    py_game = tp.Pygame(res=RES)
    screen = py_game.screen
    screen_variables_names_units = {'names': {'battery_level': 'Battery Level', 'flight_phase': 'Flight Phase',
        'auto_pilot_armed': 'Auto-Pilot Armed', 'speed': 'Speed', 'temperature': 'Temperature',
        'flight_time': 'Flight Time'},
        'units': {'battery_level': '%', 'flight_phase': '', 'auto_pilot_armed': '', 'speed': '', 'temperature': '°C',
            'flight_time': 'sec'}}
    print(py_game)

    # init drone
    drone = tp.Drone(res=RES, mirror_down=True, speed=50)
    drone.power_up()
    drone_is_one = drone.drone_is_on
    rc_values = (0, 0, 0, 0)
    img = drone.get_frame()
    img_april_tag = img
    img_obj_det = img
    landing_zone_xy, img_obj_det, _ = lz_finder.get_final_lz(img)  # first time always takes longest, so do it here
    battery_level = None
    temperature = None
    flight_time = None
    print(drone)

    # init auto_pilot
    auto_pilot = tp.Autopilot(res=RES, speed=40, apriltag_factor=2)
    auto_pilot_armed = auto_pilot.auto_pilot_armed
    prev_error = (0, 0, 0, 0)
    print(auto_pilot)
    landing_zone_xy_list = []
    NUMBER_ROLLING_XY_VALUES = 5

    while drone_is_one:

        ##############################################
        start1 = time.time()
        img = drone.get_frame()
        end1 = time.time()
        print("Frame:", end1 - start1)

        ##############################################

        start2 = time.time()
        target_xy = None
        area = 0
        rc_values = (0, 0, 0, 0)
        landing_zone_xy = None
        landing_zone_xy_avg = None

        # auto_pilot computations
        if drone.flight_phase == "Take-off":
            pass

        elif drone.flight_phase == "Hover":
            pass

        elif drone.flight_phase == "Approach":
            if auto_pilot.auto_pilot_armed:
                img_april_tag, target_xy, area = apriltag_finder.apriltag_center_area(img)
            else:
                img_april_tag, _, _ = apriltag_finder.apriltag_center_area(img)

        elif drone.flight_phase == "Landing":
            if auto_pilot.auto_pilot_armed:
                landing_zone_xy, img_obj_det, _ = lz_finder.get_final_lz(img)
            else:
                _, img_obj_det, _ = lz_finder.get_final_lz(img)

            if landing_zone_xy is not None:
                landing_zone_xy_avg, landing_zone_xy_list = rolling_average(landing_zone_xy_list, landing_zone_xy,
                                                                            NUMBER_ROLLING_XY_VALUES)  # print("Landing Zone XY:", landing_zone_xy_avg)

        if target_xy is not None:
            # send rc commands based on target_xy
            error = auto_pilot.get_alignment_error(target_xy, area)
            rc_values = auto_pilot.track_target(rc_values, target_xy, error, prev_error, area)
            prev_error = error

        # watch out for keyboard input, return (0,0,0,0) if no key is pressed
        rc_values = tp.keyboard_rc(drone, rc_values, py_game, drone.SPEED)

        # send rc commands to drone; order: 1) keyboard, 2) auto_pilot, 3) default (0,0,0,0)
        drone.me.send_rc_control(rc_values[0], rc_values[1], rc_values[2], rc_values[3])

        end2 = time.time()
        print("Control:", end2 - start2)

        ##############################################

        start3 = time.time()
        # register ESC has high priority
        if tp.exit_app_key_pressed(py_game):
            drone_is_one = False
            drone.power_down()
            cv2.destroyAllWindows()
            pygame.quit()
            exit_program = True
            break

        # switch speed
        if tp.switch_speed_key_pressed(py_game):
            if drone.SPEED == 50:
                drone.SPEED = 100
            elif drone.SPEED == 100:
                drone.SPEED = 50

        # switch auto_pilot on/off
        if tp.auto_pilot_key_pressed(py_game):
            if drone.flight_phase == ("Approach" or "Landing"):
                auto_pilot.auto_pilot_armed = not auto_pilot.auto_pilot_armed

        # set flight phase
        if tp.takeoff_phase_key_pressed(py_game):
            if drone.me.is_flying is not True:
                drone.me.takeoff()
                drone.flight_phase = "Take-off"
        elif tp.hover_phase_key_pressed(py_game):
            drone.flight_phase = "Hover"
            if auto_pilot.auto_pilot_armed == True:
                auto_pilot.auto_pilot_armed = False
        elif tp.approach_phase_key_pressed(py_game):
            drone.flight_phase = "Approach"
        elif tp.landing_phase_key_pressed(py_game):
            drone.flight_phase = "Landing"

        end3 = time.time()
        print("Keyboard:", end3 - start3)
        ##############################################

        start4 = time.time()

        # get drone sensor data for display in pygame
        battery_level, temperature, flight_time, _, _ = drone.get_drone_sensor_data()
        # video feed via pygame
        if drone.flight_phase == "Take-off":
            pass  # no video feed first, first cool down using fans
        elif drone.flight_phase == "Hover":
            py_game.display_video_feed(screen, img)
        elif drone.flight_phase == "Approach":
            py_game.display_video_feed(screen, img_april_tag)
        elif drone.flight_phase == "Landing":
            py_game.display_video_feed(screen, img_obj_det)

        if battery_level <= 15:
            py_game.display_status(screen=screen, text="Battery level low: " + str(battery_level) + "%",
                                   show_warning=True)
        else:
            py_game.display_multiple_status(screen=screen, screen_variables_names_units=screen_variables_names_units,
                                            v_pos=10, h_pos=10, battery_level=battery_level, flight_time=flight_time,
                                            temperature=temperature, speed=drone.SPEED, flight_phase=drone.flight_phase,
                                            auto_pilot_armed=auto_pilot.auto_pilot_armed)
        pygame.display.flip()

        end4 = time.time()
        print("Pygame:", end4 - start4)  ##############################################

    if exit_program == True:
        sys.exit()
