import pygame, sys, time, copy
import cv2 as cv
import numpy as np
from pygame import USEREVENT

import tello_package as tp
import vision_package as vp
import utils as ut
from config import tello_wifi

# General Parameters
WIDTH, HEIGHT = 640, 480  # (1280, 720), (640, 480), (320, 240)
FPS_FOR_VIDEO_RECORDING = 5  # realistically rather 1/250ms i.e. 4 FPS as one loop takes 250ms
TIMER_FRAME_Q_EVENT = 500  # introduces a delay of 500ms for the frame_q

# Manual RC
MANUAL_SPEED = 50

# Auto Pilot
USE_SEG_FOR_LZ = True
APRILTAG_FACTOR = 2
AUTO_PILOT_SPEED_APPROACH = 40
AUTO_PILOT_SPEED_LANDING = 10
POSITION_TOLERANCE_THRESHOLD_APPROACH = 15  # from 0 to 100;
POSITION_TOLERANCE_THRESHOLD_LANDING = 15
AUTO_TRANSITION_TIMER_APPROACH = 5  # seconds
AUTO_TRANSITION_TIMER_LANDING = 5  # seconds
PID=[0.4, 0, 0.4]

# LZ Finder
MAX_DET = None
STRIDE = 75
R_LANDING_FACTOR = 8
WEIGHT_DIST = 1
WEIGHT_RISK = 1
WEIGHT_OB = 1
NUMBER_ROLLING_XY_VALUES = 10
DRAW_LZS_IN_LZ_FINDER=False

# YOLO
LABELS_STR_LIST_BLACKLIST = ['train', 'stop sign', "dining table"]
LABELS_STR_LIST_WHITELIST = [cls for cls in vp.labels_yolo.keys() if cls not in LABELS_STR_LIST_BLACKLIST]
LABELS_DIC_FILTERED = {key: value for key, value in vp.labels_yolo.items() if key in LABELS_STR_LIST_WHITELIST}
LABELS_IDS_LIST_FILTERED = list(LABELS_DIC_FILTERED.values())
MODEL_OBJ_DET_PATH = 'vision_package/yoloV8_models/yolov8n.pt'
MODEL_SEG_PATH = 'vision_package/yoloV8_models/yolov8n-seg.pt'
YOLO_VERBOSE = True
CONF_THRES_OBJ_DET = 0.25

# Image Capture
video_recording_on = False
taking_pictures_on = False

# Helper
RES = (WIDTH, HEIGHT)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
ORANGE = (255, 165, 0)
BLUE = (0, 0, 255)
dummy_img_for_init = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
blank_img_for_takeoff_touchdown_screen = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
IMG_FOR_STARTING_SCREEN_PATH = "../data/assets/starting_screen.jpg"
R_LANDING_DEFAULT = int(WIDTH * 20 / 100)  # pixel
ALTITUDE_CM_DEFAULT = 70  # cm

#######################################################################################################################

class DroneController:

    def __init__(self):

        print('[CONTROLLER-THREAD]: DroneController __init__ started')

        # APRILTAG FINDER
        self.apriltag_finder = vp.ApriltagFinder(resolution=RES)

        # AUTO PILOT
        self.auto_pilot = tp.Autopilot(res=RES, PID=PID, auto_pilot_speed=AUTO_PILOT_SPEED_APPROACH,
                                       apriltag_factor=APRILTAG_FACTOR,
                                       position_tolerance_threshold=POSITION_TOLERANCE_THRESHOLD_APPROACH)

        # IMAGE CAPTURE
        self.image_capture = tp.ImageCapture(resolution=RES, fps_video_recording=FPS_FOR_VIDEO_RECORDING)

        # PYGAME
        self.py_game = tp.Pygame(res=RES)
        self.screen = self.py_game.screen
        img_for_starting_screen = pygame.image.load(IMG_FOR_STARTING_SCREEN_PATH)
        img_for_starting_screen = pygame.transform.scale(img_for_starting_screen, (WIDTH, HEIGHT))
        self.screen.blit(img_for_starting_screen, (0, 0))
        pygame.display.flip()
        self.py_game.set_timer(USEREVENT + 1, TIMER_FRAME_Q_EVENT)

        # LZ FINDER
        print('[CONTROLLER-THREAD]: Initialize lz_finder')
        self.lz_finder = vp.LzFinder(model_obj_det_path=MODEL_OBJ_DET_PATH, model_seg_path=MODEL_SEG_PATH,
                                     labels_dic_filtered=LABELS_DIC_FILTERED,
                                     max_det=MAX_DET, res=RES, verbose=YOLO_VERBOSE,
                                     stride=STRIDE,
                                     use_seg_for_lz=USE_SEG_FOR_LZ,
                                     weight_dist=WEIGHT_DIST, weight_risk=WEIGHT_RISK, weight_obs=WEIGHT_OB,
                                     draw_lzs=DRAW_LZS_IN_LZ_FINDER)

        _, _, _ = self.lz_finder.get_final_lz(dummy_img_for_init, R_LANDING_DEFAULT)
        print('[CONTROLLER-THREAD]: lz_finder initialized including DUMMY inference')

        # DRONE
        ut.connect_to_wifi(tello_wifi)
        self.drone = tp.Drone(res=RES, speed=MANUAL_SPEED)
        self.drone.power_up()

        print('[CONTROLLER-THREAD]: __init__ finished')

    def run(self):

        # General Variables
        exit_program = False

        # Images
        img_april_tag = None
        frame_lz_inference = dummy_img_for_init
        risk_map = dummy_img_for_init

        # Drone
        battery_level = 100
        flight_time = 0
        altitude_cm=0

        # Auto Pilot
        prev_error = (0, 0, 0, 0)
        apriltag_center_xy = (0, 0)
        area = 0
        landing_zone_xy = (int(WIDTH / 2), int(HEIGHT / 2))
        list_of_lz_tuples = []
        landing_zone_xy_avg = (int(WIDTH / 2), int(HEIGHT / 2))
        target_xy = (0, 0)
        list_of_position_clearance_timestamps = []
        seconds_within_current_clearance_period = 0
        auto_transition_timer = AUTO_TRANSITION_TIMER_APPROACH
        position_tolerance_threshold = POSITION_TOLERANCE_THRESHOLD_APPROACH

        # Pygame
        color = BLUE

        self.py_game.display_video_feed(self.screen, blank_img_for_takeoff_touchdown_screen)

        # Image Capture
        video_recording_on = False
        taking_pictures_on = False
        img_num = 1
        n_img_batch = 50
        img_saving_path = ''
        last_img_time = time.time()
        out = None

        while not exit_program:

            controller_loop_start_time = time.time()

            print('[CONTROLLER-THREAD]: Flight-Phase: ' + self.drone.flight_phase)

            rc_values = (0, 0, 0, 0)

            # NEW FRAME ################################################################################################

            frame_from_drone = self.drone.get_frame()

            # REGISTER KEYBOARD INPUT #################################################################################

            # ESC the program
            if tp.exit_app_key_pressed(self.py_game):
                self.drone.power_down()
                cv.destroyAllWindows()
                pygame.quit()
                exit_program = True
                break

            # switch speed
            if tp.switch_speed_key_pressed(self.py_game):
                if self.drone.speed == 50:
                    self.drone.speed = 100
                elif self.drone.speed == 100:
                    self.drone.speed = 50

            # switch auto_pilot on/off
            if tp.auto_pilot_key_pressed(self.py_game):

                apriltag_center_xy = (0, 0)
                list_of_position_clearance_timestamps = []
                seconds_within_current_clearance_period = 0
                list_of_lz_tuples = []
                target_xy = (0, 0)
                landing_zone_xy = (int(WIDTH / 2), int(HEIGHT / 2))
                landing_zone_xy_avg = (int(WIDTH / 2), int(HEIGHT / 2))
                prev_error = (0, 0, 0, 0)
                area = 0
                print('[CONTROLLER-THREAD]: auto_pilot_key_pressed')

                if self.drone.flight_phase == "Approach":
                    self.auto_pilot.auto_pilot_armed = not self.auto_pilot.auto_pilot_armed
                    print('[CONTROLLER-THREAD]: auto_pilot_armed = ', self.auto_pilot.auto_pilot_armed)
                elif self.drone.flight_phase == "Landing":
                    self.auto_pilot.auto_pilot_armed = not self.auto_pilot.auto_pilot_armed
                    print('[CONTROLLER-THREAD]: auto_pilot_armed = ', self.auto_pilot.auto_pilot_armed)

            # set flight phase
            if tp.takeoff_phase_key_pressed(self.py_game):
                self.drone.flight_phase = "Take-off"
                self.py_game.display_video_feed(self.screen, blank_img_for_takeoff_touchdown_screen)
                if self.drone.me.is_flying is not True:
                    self.drone.me.takeoff()

            elif tp.hover_phase_key_pressed(self.py_game):
                self.drone.flight_phase = "Hover"
                print('[CONTROLLER-THREAD]: Hover phase +++++++++++++++++')
                if self.auto_pilot.auto_pilot_armed:
                    self.auto_pilot.auto_pilot_armed = False

            elif tp.obj_det_phase_key_pressed(self.py_game):
                self.drone.flight_phase = "YOLO Object Detection"
                print('[CONTROLLER-THREAD]: YOLO Object Detection +++++++++++++++++')

            elif tp.seg_phase_key_pressed(self.py_game):
                self.drone.flight_phase = "YOLO Segmentation"
                print('[CONTROLLER-THREAD]: YOLO Segmentation +++++++++++++++++')

            elif tp.risk_map_key_pressed(self.py_game):
                self.drone.flight_phase = "Risk Map"
                print('[CONTROLLER-THREAD]: Risk Map +++++++++++++++++')

            elif tp.approach_phase_key_pressed(self.py_game):
                self.drone.flight_phase = "Approach"
                self.auto_pilot.autopilot_speed = AUTO_PILOT_SPEED_APPROACH
                print('[CONTROLLER-THREAD]: Approach phase +++++++++++++++++')

            elif tp.landing_phase_key_pressed(self.py_game):
                self.drone.flight_phase = "Landing"
                self.auto_pilot.autopilot_speed = AUTO_PILOT_SPEED_LANDING
                list_of_lz_tuples = []
                print('[CONTROLLER-THREAD]: Landing phase +++++++++++++++++')

            # control image capture
            if tp.taking_pictures_key_pressed(self.py_game):
                taking_pictures_on = not taking_pictures_on

            if tp.recording_video_key_pressed(self.py_game):
                video_recording_on = not video_recording_on

            # AUTO-PILOT COMPUTATIONS #################################################################################
            if self.drone.flight_phase == "Take-off":
                pass

            elif self.drone.flight_phase == "Hover":
                pass

            elif self.drone.flight_phase == "Approach":

                img_april_tag, apriltag_center_xy, area = self.apriltag_finder.apriltag_center_area(frame_from_drone)

                if self.auto_pilot.auto_pilot_armed:
                    target_xy = apriltag_center_xy
                else:
                    img_april_tag, _, _ = self.apriltag_finder.apriltag_center_area(frame_from_drone)

            elif self.drone.flight_phase == "Landing" or self.drone.flight_phase == "Risk Map":

                frame_lz_inference = copy.deepcopy(frame_from_drone)  # otherwise frame_from_drone is overwritten

                battery_level, temperature, flight_time, _, altitude_cm = self.drone.get_drone_sensor_data()

                r_landing = int(R_LANDING_DEFAULT * ALTITUDE_CM_DEFAULT/altitude_cm)

                landing_zone_xy, frame_lz_inference, risk_map = self.lz_finder.get_final_lz(frame_lz_inference, r_landing)

                landing_zone_xy_avg, list_of_lz_tuples = ut.rolling_average_of_tuples(list_of_tuples=list_of_lz_tuples,
                                                                                      new_tuple=landing_zone_xy,
                                                                                      number_of_values=NUMBER_ROLLING_XY_VALUES)
                area = None

                if self.auto_pilot.auto_pilot_armed:
                    if len(list_of_lz_tuples) >= NUMBER_ROLLING_XY_VALUES:
                        target_xy = landing_zone_xy_avg

            elif self.drone.flight_phase == "YOLO Object Detection":
                pass

            elif self.drone.flight_phase == "YOLO Segmentation":
                pass

            elif self.drone.flight_phase == "Touch-down":
                list_of_position_clearance_timestamps = []
                seconds_within_current_clearance_period = 0

            timer_auto_transition = round(5 - seconds_within_current_clearance_period, 1)
            if timer_auto_transition < 0:
                timer_auto_transition = 0

            # EXECUTE CONTROL ACTIONS #################################################################################

            if target_xy != (0, 0):
                error, position_within_tolerance = self.auto_pilot.get_alignment_error(target_xy, area)
                print('[CONTROLLER-THREAD]: error: ', error)
                rc_values = self.auto_pilot.track_target(rc_values, target_xy, error, prev_error, area)
                prev_error = error

                # AUTO TRANSITION TIMER #################################################################################
                newest_timestamp = time.time()
                if position_within_tolerance:
                    list_of_position_clearance_timestamps.append((newest_timestamp, position_within_tolerance))
                    seconds_within_current_clearance_period = ut.seconds_within_current_clearance_period(
                        list_of_position_clearance_timestamps, newest_timestamp)
                else:
                    list_of_position_clearance_timestamps = []
                    seconds_within_current_clearance_period = 0

                if self.drone.flight_phase == "Approach":
                    auto_transition_timer = AUTO_TRANSITION_TIMER_APPROACH
                    self.auto_pilot.position_tolerance_threshold = POSITION_TOLERANCE_THRESHOLD_APPROACH

                elif self.drone.flight_phase == "Landing":
                    auto_transition_timer = AUTO_TRANSITION_TIMER_LANDING
                    self.auto_pilot.position_tolerance_threshold = POSITION_TOLERANCE_THRESHOLD_LANDING

                if seconds_within_current_clearance_period >= auto_transition_timer:
                    if self.drone.flight_phase == "Approach":
                        list_of_position_clearance_timestamps = []
                        seconds_within_current_clearance_period = 0
                        error = (0, 0, 0, 0)
                        prev_error = (0, 0, 0, 0)
                        self.drone.me.move_down(30)
                        self.auto_pilot.autopilot_speed = AUTO_PILOT_SPEED_LANDING
                        self.drone.flight_phase = "Landing"
                    elif self.drone.flight_phase == "Landing":
                        self.drone.flight_phase = "Touch-down"
                        self.auto_pilot.auto_pilot_armed = False
                        self.drone.me.land()
                        list_of_position_clearance_timestamps = []
                        seconds_within_current_clearance_period = 0
                        list_of_lz_tuples = []
                        target_xy = (0, 0)
                        landing_zone_xy = (int(WIDTH / 2), int(HEIGHT / 2))
                        landing_zone_xy_avg = (int(WIDTH / 2), int(HEIGHT / 2))
                        prev_error = (0, 0, 0, 0)
                        area = 0

            rc_values = tp.keyboard_rc(self.drone, rc_values, self.py_game, self.drone.speed)

            self.drone.me.send_rc_control(rc_values[0], rc_values[1], rc_values[2], rc_values[3])

            # PYGAME VIDEO ##########################################################################################

            if self.drone.flight_phase == "Take-off":
                pass

            elif self.drone.flight_phase == "Hover":
                frame_hover = ut.add_central_dot(frame_from_drone)
                self.py_game.display_video_feed(self.screen, frame_hover)


            elif self.drone.flight_phase == "YOLO Object Detection":
                results_obj_det = self.lz_finder.object_detector.model.predict(source=frame_from_drone,
                                                                               verbose=YOLO_VERBOSE,
                                                                               classes=LABELS_IDS_LIST_FILTERED)
                frame_obj_det_annotated = results_obj_det[0].plot()
                frame_obj_det_annotated = ut.add_central_dot(frame_obj_det_annotated)
                self.py_game.display_video_feed(self.screen, frame_obj_det_annotated)

            elif self.drone.flight_phase == "YOLO Segmentation":
                results_seg_engine = self.lz_finder.seg_engine.model.predict(source=frame_from_drone,
                                                                             verbose=YOLO_VERBOSE,
                                                                             classes=LABELS_IDS_LIST_FILTERED)
                frame_seg_annotated = results_seg_engine[0].plot()
                frame_seg_annotated = ut.add_central_dot(frame_seg_annotated)
                self.py_game.display_video_feed(self.screen, frame_seg_annotated)

            elif self.drone.flight_phase == "Risk Map":
                risk_map = cv.applyColorMap(risk_map, cv.COLORMAP_BONE)
                self.py_game.display_video_feed(self.screen, risk_map)

            elif self.drone.flight_phase == "Approach":
                img_april_tag = ut.add_central_dot(img_april_tag)
                img_april_tag = ut.add_central_dot(img_april_tag)
                self.py_game.display_video_feed(self.screen, img_april_tag)

            elif self.drone.flight_phase == "Landing":
                frame_lz_inference = ut.add_central_dot(frame_lz_inference)

                if timer_auto_transition < 2:
                    color = RED

                frame_lz_inference = cv.circle(frame_lz_inference, landing_zone_xy_avg, self.lz_finder.r_landing, color,
                                               3)
                frame_lz_inference = cv.circle(frame_lz_inference, landing_zone_xy, self.lz_finder.r_landing, color, 1)

                self.py_game.display_video_feed(self.screen, frame_lz_inference)

            elif self.drone.flight_phase == "Touch-down":
                img_for_starting_screen = pygame.transform.scale(img_for_starting_screen, (WIDTH, HEIGHT))
                self.screen.blit(img_for_starting_screen, (0, 0))
                pygame.display.flip()

            # DISPLAY STATUS

            self.py_game.display_multiple_status(screen=self.screen, v_pos=10, h_pos=10, battery_level=battery_level,
                                                 flight_time=flight_time,  # temperature=temperature,
                                                 altitude_cm=altitude_cm,  # speed=self.drone.speed,
                                                 auto_pilot_speed=self.auto_pilot.autopilot_speed,
                                                 flight_phase=self.drone.flight_phase,
                                                 auto_pilot_armed=self.auto_pilot.auto_pilot_armed,
                                                 timer_auto_transition=timer_auto_transition,
                                                 # taking_pictures_on=taking_pictures_on,
                                                 # video_recording_on=video_recording_on,
                                                 )
            if battery_level <= 15:
                self.py_game.display_status(screen=self.screen, text="Battery level low: " + str(battery_level) + "%",
                                            show_warning=True)
            else:
                pass

            pygame.display.flip()

            # IMAGE CAPTURE ###########################################################################################

            if taking_pictures_on == True:
                if time.time() >= last_img_time + 1:
                    img_num, img_saving_path, last_img_time = self.image_capture.save_image(frame_from_drone, img_num,
                                                                                            img_saving_path)
                    print('[CONTROLLER-THREAD]: Picture saved to disk')
                if img_num > n_img_batch:
                    img_num = 1
                    taking_pictures_on = False
                    print('[CONTROLLER-THREAD]: All Pictures saved to disk')

            if video_recording_on == True:
                out = self.image_capture.record_video(frame_from_drone, out)
            else:
                if out is not None:
                    out.release()
                    out = None

            # TIME TRACKING ###########################################################################################

            controller_loop_end_time = time.time()
            ut.print_interval_ms("[CONTROLLER-THREAD]: Loop Time", controller_loop_start_time, controller_loop_end_time)
            ut.print_fps("[CONTROLLER-THREAD]: FPS", controller_loop_start_time, controller_loop_end_time)
            print()
            print()

            # time.sleep(1 / FPS)

        if exit_program:
            sys.exit()


def main():
    yolo_drone = DroneController()
    yolo_drone.run()


if __name__ == '__main__':
    main()
