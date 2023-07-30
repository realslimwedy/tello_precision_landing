#######################################################################################################################

# TODOS
# - [ ] Try another FPS
# - [ ] fix warning message to be removed properly
# - [ ] Add initial slide as startup screen in pygame
# - [ ] Try to use frame as global variable not as queue
# - [ ] Try limiting objects detected to only one
# - [ ] Display FPS/Time of each thread
# - [ ] Scale circle size with drone height
# - [ ] place initial variables as far on top as possible

# - [ ] show how much % CPU and memory each thread uses

# options: optimize classes and detection and only use one main thread
# second option: have predictions done in yolo thread and pick up from queue

# - [ ] Add option to show object detection in GUI
# - [ ] Add option to show segmentation mask in GUI
# - [ ] Add image capturing capability
# - [ ] Add logging


# IMPORTS #############################################################################################################

import pygame, sys, time, threading, queue, copy
import cv2 as cv
import numpy as np
from pygame import USEREVENT

import tello_package as tp
import vision_package as vp
import utils as ut

# SENSITIVE PARAMETERS ################################################################################################

FPS = 25
TIMER_FRAME_Q_EVENT = 500  # introduces a delay of 500ms for the frame_q
NUMBER_ROLLING_XY_VALUES = 10
AUTO_PILOT_SPEED_APPROACH = 40
AUTO_PILOT_SPEED_LANDING = 10
POSITION_TOLERANCE_THRESHOLD = 20  # from 0 to 100; [CONTROLLER-THREAD]: error:  (17.1875, 0.8333333333333334, -15.769675925925927, 0)
AUTO_TRANSITION_TIMER_APPROACH = 5  # seconds
AUTO_TRANSITION_TIMER_LANDING = 10  # seconds
R_LANDING_FACTOR = 8
STRIDE = 75
WEIGHT_DIST = 1
WEIGHT_RISK = 0
WEIGHT_OB = 1

#######################################################################################################################

WIDTH, HEIGHT = 640, 480  # (1280, 720), (640, 480), (320, 240)
RES = (WIDTH, HEIGHT)

mirror_down = True
SPEED = 50
APRILTAG_FACTOR = 2

RED = (255, 0, 0)
GREEN = (0, 255, 0)
ORANGE = (255, 165, 0)

exit_program = False

dummy_img_for_init = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
blank_img_for_takeoff_screen = np.zeros((RES[1], RES[0], 3), dtype=np.uint8)

labels_str_list_blacklist = ['train', 'stop sign', 'bottle', 'carrot', "dining table"]
labels_str_list_whitelist = [cls for cls in vp.labels_yolo.keys() if cls not in labels_str_list_blacklist]
labels_dic_filtered = {key: value for key, value in vp.labels_yolo.items() if key in labels_str_list_whitelist}
labels_ids_list_filtered = list(labels_dic_filtered.values())

model_obj_det_path = 'vision_package/yoloV8_models/yolov8n.pt'
model_seg_path = 'vision_package/yoloV8_models/yolov8n-seg.pt'
yolo_verbose = True
use_seg_for_lz = False
max_det = None
r_landing_factor = 8
stride = 75

video_recording_on = False
taking_pictures_on = False

frame_q = queue.Queue()
img_obj_det_q = queue.Queue()
landing_zone_xy_q = queue.Queue()

controller_initialized = threading.Event()


#######################################################################################################################

def yolo_thread():
    print('[YOLO-THREAD]: __init__ started')

    global exit_program, landing_zone_xy_q  # , frame_q, img_obj_det_q

    frame_from_queue = None
    print('[YOLO-THREAD]: yolo_thread fully initialized')

    controller_initialized.wait()

    while not exit_program:
        yolo_start_time = time.time()

        try:
            frame_from_queue = frame_q.get()
        except queue.Empty:
            print('[YOLO-THREAD]: EXCEPTION frame_queue is empty')
            pass

        if frame_from_queue is not None:
            print('[YOLO-THREAD]: starting inference')
            yolo_inference_start_time = time.time()

            landing_zone_xy, img_from_lz_finder, _ = lz_finder.get_final_lz(frame_from_queue)

            yolo_inference_end_time = time.time()
            ut.print_interval_ms('[YOLO-THREAD]: Inference Time', yolo_inference_start_time, yolo_inference_end_time)

        else:
            print('[YOLO-THREAD]: frame_from_queue is None')
            pass

        if img_from_lz_finder is not None:
            print(f'[YOLO-THREAD]: Landing Zone: {landing_zone_xy}')
            with landing_zone_xy_q.mutex:
                landing_zone_xy_q.queue.clear()
            landing_zone_xy_q.put(landing_zone_xy)
            print('[YOLO-THREAD]: landing_zone_xy put in queue')
        else:
            print('[YOLO-THREAD]: landing_zone_xy is None')

        if img_from_lz_finder is not None:
            print(f'[YOLO-THREAD]: img_from_lz_finder: {img_from_lz_finder.shape}')
            with img_obj_det_q.mutex:
                img_obj_det_q.queue.clear()
            img_obj_det_q.put(img_from_lz_finder)
            print('[YOLO-THREAD]: img_from_lz_finder put in queue')
        else:
            print('[YOLO-THREAD]: img_from_lz_finder is None')

        frame_from_queue = None

        yolo_end_time = time.time()
        ut.print_interval_ms('[YOLO-THREAD]: Loop Time', yolo_start_time, yolo_end_time)


#######################################################################################################################

class DroneController:

    def __init__(self):

        print('[CONTROLLER-THREAD]: __init__ started')

        # PYGAME
        self.py_game = tp.Pygame(res=RES)
        self.screen = self.py_game.screen
        img_for_starting_screen_path = "../data/assets/starting_screen.jpg"
        img_for_starting_screen = pygame.image.load(img_for_starting_screen_path)
        img_for_starting_screen = pygame.transform.scale(img_for_starting_screen, (WIDTH, HEIGHT))
        self.screen.blit(img_for_starting_screen, (0, 0))
        pygame.display.flip()
        self.py_game.set_timer(USEREVENT + 1, TIMER_FRAME_Q_EVENT)

        # DRONE
        self.drone = tp.Drone(res=RES, mirror_down=mirror_down, speed=SPEED)
        self.drone.power_up()

        # LZ FINDER
        print('[CONTROLLER-THREAD]: Initialize lz_finder')
        self.lz_finder = vp.LzFinder(model_obj_det_path=model_obj_det_path, model_seg_path=model_seg_path,
                                     labels_dic_filtered=labels_dic_filtered, max_det=max_det, res=RES,
                                     use_seg_for_lz=use_seg_for_lz, r_landing_factor=r_landing_factor, stride=stride,
                                     verbose=True, weightDist=WEIGHT_DIST, weightRisk=WEIGHT_RISK, weightOb=WEIGHT_OB,
                                     draw_lzs=False)
        _, _, _ = self.lz_finder.get_final_lz(dummy_img_for_init)
        print('[CONTROLLER-THREAD]: lz_finder initialized inclunding DUMMY inference')

        # APRILTAG FINDER
        self.apriltag_finder = vp.ApriltagFinder(resolution=RES)

        # AUTO PILOT
        self.auto_pilot = tp.Autopilot(res=RES, auto_pilot_speed=AUTO_PILOT_SPEED_APPROACH,
                                       apriltag_factor=APRILTAG_FACTOR,
                                       position_tolerance_threshold=POSITION_TOLERANCE_THRESHOLD)

        # IMAGE CAPTURE
        self.image_capture = tp.ImageCapture(RES, FPS)

        print('[CONTROLLER-THREAD]: __init__ finished')

        controller_initialized.set()

    def run(self):
        print('[CONTROLLER-THREAD]: run() method started')

        global exit_program, prev_error, landing_zone_xy_q  # , frame_q, img_obj_det_q

        prev_error = (0, 0, 0, 0)
        img_april_tag = None
        frame_lz_inference = dummy_img_for_init
        apriltag_center_xy = (0, 0)
        list_of_lz_tuples = []
        list_of_position_clearance_timestamps = []
        seconds_within_current_clearance_period = 0
        target_xy = (0, 0)
        landing_zone_xy = (0, 0)
        landing_zone_xy_avg = (0, 0)
        area = 0
        AUTO_TRANSITION_TIMER = AUTO_TRANSITION_TIMER_APPROACH
        self.py_game.display_video_feed(self.screen, blank_img_for_takeoff_screen)
        out = None
        img_num = 1
        n_img_batch = 50
        img_saving_path = ''
        last_img_time = time.time()
        img_saving_path = '../data/images_saved_by_drone'
        video_recording_path = '../data/videos_saved_by_drone'

        while not exit_program:

            controller_loop_start_time = time.time()
            print('[CONTROLLER-THREAD]: run() LOOP started')
            print(f'[CONTROLLER-THREAD]: Number of active Threads = {threading.active_count()}')

            for thread in threading.enumerate():
                print(f"[CONTROLLER-THREAD]: Thread Name: {thread.name}")

            rc_values = (0, 0, 0, 0)
            img_obj_det_from_queue = None

            # NEW FRAME ################################################################################################

            frame_from_drone = self.drone.get_frame()
            print('[CONTROLLER-THREAD]: frame_from_drone received from drone.get_frame()')

            # FRAME TO QUEUE EVERY 500 ms ##############################################################################

            '''for event in pygame.event.get():
                if event.type == USEREVENT + 1:
                    with frame_q.mutex:
                        frame_q.queue.clear()
                    frame_q.put(frame_from_drone)
                    print('[CONTROLLER-THREAD]: frame_from_drone put in queue')'''

            '''if event.type == USEREVENT + 2:
                rc_values = tp.keyboard_rc(self.drone, rc_values, self.py_game, self.drone.speed)
                self.drone.me.send_rc_control(rc_values[0], rc_values[1], rc_values[2], rc_values[3])'''

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
                landing_zone_xy = (0, 0)
                landing_zone_xy_avg = (0, 0)
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
                self.py_game.display_video_feed(self.screen, blank_img_for_takeoff_screen)
                if self.drone.me.is_flying is not True:
                    self.drone.me.takeoff()

            elif tp.hover_phase_key_pressed(self.py_game):
                self.drone.flight_phase = "Hover"
                print('[CONTROLLER-THREAD]: Hover phase +++++++++++++++++')
                if self.auto_pilot.auto_pilot_armed:
                    self.auto_pilot.auto_pilot_armed = False

            elif tp.approach_phase_key_pressed(self.py_game):
                self.drone.flight_phase = "Approach"
                self.auto_pilot.autopilot_speed = AUTO_PILOT_SPEED_APPROACH
                print('[CONTROLLER-THREAD]: Approach phase +++++++++++++++++')

            elif tp.landing_phase_key_pressed(self.py_game):
                self.drone.flight_phase = "Landing"
                self.auto_pilot.autopilot_speed = AUTO_PILOT_SPEED_LANDING
                list_of_lz_tuples = []
                print('[CONTROLLER-THREAD]: Landing phase +++++++++++++++++')

            elif tp.exploration_obj_det_phase_key_pressed(self.py_game):
                self.drone.flight_phase = "Exploration Obj-Det"
                print('[CONTROLLER-THREAD]: Exploration Object Detection +++++++++++++++++')

            elif tp.exploration_seg_phase_key_pressed(self.py_game):
                self.drone.flight_phase = "Exploration Seg"
                print('[CONTROLLER-THREAD]: Exploration Segmentation +++++++++++++++++')

            elif tp.exploration_lz_finder_key_pressed(self.py_game):
                self.drone.flight_phase = "Exploration LZ-Finder"
                print('[CONTROLLER-THREAD]: Exploration LZ-Finder +++++++++++++++++')

            # control image capture
            if tp.taking_pictures_key_pressed(self.py_game):
                taking_pictures_on = not taking_pictures_on

            if tp.recording_video_key_pressed(self.py_game):
                recording_on = not recording_on


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

            elif self.drone.flight_phase == "Landing":

                frame_lz_inference = copy.deepcopy(frame_from_drone)  # otherwise frame_from_drone is overwritten

                landing_zone_xy, frame_lz_inference, _ = self.lz_finder.get_final_lz(frame_lz_inference)

                landing_zone_xy_avg, list_of_lz_tuples = ut.rolling_average_of_tuples(list_of_tuples=list_of_lz_tuples,
                                                                                      new_tuple=landing_zone_xy,
                                                                                      number_of_values=NUMBER_ROLLING_XY_VALUES)
                area = None

                if self.auto_pilot.auto_pilot_armed:
                    if len(list_of_lz_tuples) >= NUMBER_ROLLING_XY_VALUES:
                        target_xy = landing_zone_xy_avg

            elif self.drone.flight_phase == "Exploration Obj-Det":
                pass
            elif self.drone.flight_phase == "Exploration Seg":
                pass
            elif self.drone.flight_phase == "Exploration LZ-Finder":
                pass

            elif self.drone.flight_phase == "Touch-down":
                list_of_position_clearance_timestamps = []
                seconds_within_current_clearance_period = 0

            '''try:
                img_obj_det_from_queue = img_obj_det_q.get_nowait()
            except queue.Empty:
                print('[CONTROLLER-THREAD]: queue is empty')

            # if self.auto_pilot.auto_pilot_armed:
            try:
                landing_zone_xy = landing_zone_xy_q.get_nowait()
            except queue.Empty:
                print('[CONTROLLER-THREAD]: landing_zone_xy_queue is empty')'''

            '''elif self.drone.flight_phase == "Landing":
                try:
                    img_obj_det = yolo_img_q.get_nowait()
                    #img_obj_det = yolo_img_queue.get_nowait()
                except queue.Empty:
                    print('[CONTROLLER-THREAD]: queue is empty')
                    continue

                if self.auto_pilot.auto_pilot_armed:
                    try:
                        landing_zone_xy = xy_landing_zone_q.get_nowait()
                        #landing_zone_xy = xy_landing_zone_queue.get_nowait()
                    except queue.Empty:
                        print('[CONTROLLER-THREAD]: queue is empty')
                        continue
                else:
                    pass'''

            # EXECUTE CONTROL ACTIONS #################################################################################

            if target_xy is not None:
                error, position_within_tolerance = self.auto_pilot.get_alignment_error(target_xy, area)
                print('[CONTROLLER-THREAD]: error: ', error)
                rc_values = self.auto_pilot.track_target(rc_values, target_xy, error, prev_error, area)
                prev_error = error

                # AUTO TRANSITION TIMER #################################################################################
                newest_timestamp = time.time()
                if position_within_tolerance:
                    print('[CONTROLLER-THREAD]: position_within_tolerance: {position_within_tolerance}')
                    list_of_position_clearance_timestamps.append((newest_timestamp, position_within_tolerance))
                    seconds_within_current_clearance_period = ut.seconds_within_current_clearance_period(
                        list_of_position_clearance_timestamps, newest_timestamp)
                else:
                    list_of_position_clearance_timestamps = []
                    seconds_within_current_clearance_period = 0

                print('[CONTROLLER-THREAD]: SECONDS WITHIN TOLERANCE: ',
                      round(seconds_within_current_clearance_period, 1))

                if self.drone.flight_phase == "Approach":
                    AUTO_TRANSITION_TIMER = AUTO_TRANSITION_TIMER_APPROACH
                elif self.drone.flight_phase == "Landing":
                    AUTO_TRANSITION_TIMER = AUTO_TRANSITION_TIMER_LANDING

                if seconds_within_current_clearance_period >= AUTO_TRANSITION_TIMER:
                    if self.drone.flight_phase == "Approach":
                        list_of_position_clearance_timestamps = []
                        seconds_within_current_clearance_period = 0
                        error = (0, 0, 0, 0)
                        prev_error = (0, 0, 0, 0)
                        self.drone.me.move_down(20)
                        self.drone.flight_phase = "Landing"
                    elif self.drone.flight_phase == "Landing":
                        self.drone.flight_phase = "Touch-down"
                        self.auto_pilot.auto_pilot_armed = False
                        self.drone.me.land()

            rc_values = tp.keyboard_rc(self.drone, rc_values, self.py_game, self.drone.speed)

            self.drone.me.send_rc_control(rc_values[0], rc_values[1], rc_values[2], rc_values[3])

            # PYGAME VIDEO ##########################################################################################

            battery_level, temperature, flight_time, _, distance_tof = self.drone.get_drone_sensor_data()

            if self.drone.flight_phase == "Take-off":
                pass

            elif self.drone.flight_phase == "Hover":
                frame_hover = ut.add_central_dot(frame_from_drone)
                self.py_game.display_video_feed(self.screen, frame_hover)

            elif self.drone.flight_phase == "Approach":
                img_april_tag = ut.add_central_dot(img_april_tag)
                img_april_tag = ut.add_central_dot(img_april_tag)
                self.py_game.display_video_feed(self.screen, img_april_tag)

            elif self.drone.flight_phase == "Landing":
                frame_lz_inference = ut.add_central_dot(frame_lz_inference)
                frame_lz_annotated = cv.circle(frame_lz_annotated, landing_zone_xy, self.lz_finder.r_landing, ORANGE, 3)
                frame_lz_annotated = cv.circle(frame_lz_annotated, landing_zone_xy_avg, self.lz_finder.r_landing, GREEN,
                                               2)

                self.py_game.display_video_feed(self.screen, frame_lz_inference)

            elif self.drone.flight_phase == "Exploration Seg":
                results_seg_engine = self.lz_finder.seg_engine.model.predict(source=frame_from_drone,
                                                                             verbose=yolo_verbose,
                                                                             classes=labels_ids_list_filtered)
                frame_seg_annotated = results_seg_engine[0].plot()
                frame_seg_annotated = ut.add_central_dot(frame_seg_annotated)
                self.py_game.display_video_feed(self.screen, frame_seg_annotated)

                # TODO: check interference with other threads  # TODO: where to do the seg inference operation, in which thread?

            elif self.drone.flight_phase == "Exploration Obj-Det":
                results_obj_det = self.lz_finder.object_detector.model.predict(source=frame_from_drone,
                                                                               verbose=yolo_verbose,
                                                                               classes=labels_ids_list_filtered)
                frame_obj_det_annotated = results_obj_det[0].plot()
                frame_obj_det_annotated = ut.add_central_dot(frame_obj_det_annotated)
                self.py_game.display_video_feed(self.screen, frame_obj_det_annotated)
            ###############################################################################################################

            elif self.drone.flight_phase == "Exploration LZ-Finder":

                frame_lz_annotated = ut.add_central_dot(frame_from_drone)

                frame_lz_annotated = cv.circle(frame_lz_annotated, landing_zone_xy, self.lz_finder.r_landing, ORANGE, 3)
                frame_lz_annotated = cv.circle(frame_lz_annotated, landing_zone_xy_avg, self.lz_finder.r_landing, GREEN,
                                               2)

                self.py_game.display_video_feed(self.screen, frame_lz_annotated)

            ###############################################################################################################

            '''elif self.drone.flight_phase == "Landing":
                if img_obj_det_from_queue is not None:
                    self.py_game.display_video_feed(self.screen, img_obj_det_from_queue)
                else:
                    print('[CONTROLLER-THREAD]: img_obj_det_from_queue is None')'''

            print('[CONTROLLER-THREAD]: Flight-Phase: ' + self.drone.flight_phase)

            # DISPLAY STATUS
            timer_auto_transition = round(5 - seconds_within_current_clearance_period, 1)
            self.py_game.display_multiple_status(screen=self.screen, v_pos=10, h_pos=10,
                                                 battery_level=battery_level, flight_time=flight_time,
                                                 temperature=temperature,
                                                 distance_tof=distance_tof,
                                                 speed=self.drone.speed,
                                                 flight_phase=self.drone.flight_phase,
                                                 auto_pilot_armed=self.auto_pilot.auto_pilot_armed,
                                                 timer_auto_transition=timer_auto_transition,
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
                    img_num, img_saving_path, last_time = self.image_capture.save_image(frame_from_drone, img_num,
                                                                                        img_saving_path)
                    print('[CONTROLLER-THREAD]: Picture saved to disk')
                if img_num > n_img_batch:
                    img_num = 1
                    taking_pictures_on = False
                    print('[CONTROLLER-THREAD]: All Pictures saved to disk')



            if recording_on == True:
                out = self.image_capture.record_video(frame_from_drone, out)
            else:
                if out is not None:
                    out.release()
                    out = None

            ############################################################################################################

            controller_loop_end_time = time.time()
            ut.print_interval_ms("[CONTROLLER-THREAD]: Loop Time", controller_loop_start_time, controller_loop_end_time)
            print()
            print()

            time.sleep(1 / FPS)

        if exit_program:
            sys.exit()


def main():
    '''thread_yolo = threading.Thread(target=yolo_thread, args=())
    thread_yolo.daemon = True
    thread_yolo.start()'''

    yolo_drone = DroneController()
    yolo_drone.run()


if __name__ == '__main__':
    main()
