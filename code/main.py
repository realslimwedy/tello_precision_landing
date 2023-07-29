import queue
# from collections import deque
import threading
import pygame, sys, time
import cv2 as cv
from ultralytics import YOLO
import tello_package as tp
import vision_package as ot
import utils as ut
from pygame import USEREVENT
import numpy as np
import djitellopy as tello

#######################################################################################################################

# TODOS
# - [ ] Try another FPS
# - [ ] Try to use frame as global variable not as queue
# - [ ] Try limiting objects detected to only one

# - [ ] Add option to show object detection in GUI
# - [ ] Add option to show segmentation mask in GUI
# - [ ] Add image capturing capability
# - [ ] Add logging

#######################################################################################################################

WIDTH, HEIGHT = 640, 480  # (1280, 720), (640, 480), (320, 240)
R_LANDING_FACTOR = 8
STRIDE = 75
FPS = 25
NUMBER_ROLLING_XY_VALUES = 5
SPEED = 50
AUTO_PILOT_SPEED = 40
APRILTAG_FACTOR = 2

RES = (WIDTH, HEIGHT)
exit_program = False

dummy_img_for_init = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

labels_str_list_blacklist = ['train', 'stop sign', 'bottle', 'carrot', "dining table"]
labels_str_list_whitelist = [cls for cls in ot.labels_yolo.keys() if cls not in labels_str_list_blacklist]
labels_dic_filtered = {key: value for key, value in ot.labels_yolo.items() if key in labels_str_list_whitelist}

model_obj_det_path = 'vision_package/yoloV8_models/yolov8n.pt'
model_seg_path = 'vision_package/yoloV8_models/yolov8n-seg.pt'

# frame_global = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

flight_phase = 'Pre-Flight'

frame_q = queue.Queue()
img_obj_det_q = queue.Queue()
landing_zone_xy_q = queue.Queue()

'''frame_q = deque(maxlen=1)
landing_zone_xy_q = deque(maxlen=1)
img_obj_det_q = deque(maxlen=1)'''

controller_initialized = threading.Event()


#######################################################################################################################


def yolo_thread():
    print('[YOLO-THREAD]: Thread started')
    global lz_finder, img_obj_det, exit_program, img_obj_det_q, landing_zone_xy_q, frame_q  # , frame_global

    # load models
    model_obj_det = YOLO(model_obj_det_path)
    print("[YOLO-THREAD]: Object Detection Model loaded")
    model_seg = YOLO(model_seg_path)

    lz_finder = LzFinder(model_obj_det=model_obj_det_path, model_seg=model_seg_path,
                         res=(640, 480), r_landing_factor=10, stride=100,
                         labels_dic_filtered=labels_dic_filtered,
                         use_seg=True)


    landing_zone_xy, img_obj_det, _ = lz_finder.get_final_lz(dummy_img_for_init)
    print('[YOLO-THREAD]: yolo_thread fully initialized')

    frame_global = None

    controller_initialized.wait()

    while True:
        yolo_start_time = time.time()

        '''try:
            frame_global = frame_q[-1]
        except IndexError:
            print('[YOLO-THREAD]: frame_queue is empty')
            pass'''

        try:
            frame_global = frame_q.get()  # frame_global = frame_queue.get()
        except queue.Empty:
            print('[YOLO-THREAD]: frame_queue is empty')
            pass

        if frame_global is not None:
            print('[YOLO-THREAD]: starting inference')
            yolo_inference_start_time = time.time()
            landing_zone_xy, img_obj_det, _ = lz_finder.get_final_lz(frame_global)
            yolo_inference_end_time = time.time()
            print('[YOLO-THREAD]: finished inference: ' + str(
                (yolo_inference_end_time - yolo_inference_start_time) * 1000))
        else:
            print('[YOLO-THREAD]: frame_global is None')
            pass

        if landing_zone_xy is not None:
            print(f'[YOLO-THREAD]: Landing Zone: {landing_zone_xy}')
            # landing_zone_xy_q.append(landing_zone_xy)
            with landing_zone_xy_q.mutex:
                landing_zone_xy_q.queue.clear()
            landing_zone_xy_q.put(landing_zone_xy)
            print('[YOLO-THREAD]: landing_zone_xy put in queue')
        else:
            print('[YOLO-THREAD]: landing_zone_xy is None')

        if img_obj_det is not None:
            print(f'[YOLO-THREAD]: img_obj_det: {img_obj_det.shape}')
            # img_obj_det_q.append(img_obj_det)
            with img_obj_det_q.mutex:
                img_obj_det_q.queue.clear()
            img_obj_det_q.put(img_obj_det)
            print('[YOLO-THREAD]: img_obj_det put in queue')
        else:
            print('[YOLO-THREAD]: img_obj_det is None')

        frame_global = None

        yolo_end_time = time.time()
        print('[YOLO-THREAD]: Loop Time: ' + str((yolo_end_time - yolo_start_time) * 1000))


#######################################################################################################################


class DroneController:

    def __init__(self):

        print('[CONTROLLER-THREAD]: init started')
        # initialize drone, py_game, auto_pilot
        self.drone = tp.Drone(res=RES, mirror_down=True, speed=SPEED)
        self.drone.power_up()

        self.apriltag_finder = ot.ApriltagFinder(resolution=RES)

        self.py_game = tp.Pygame(res=RES)
        self.screen = self.py_game.screen

        self.auto_pilot = tp.Autopilot(res=RES, auto_pilot_speed=AUTO_PILOT_SPEED, apriltag_factor=APRILTAG_FACTOR)

        # initialize variables

        self.py_game.set_timer(USEREVENT + 1, 50)
        self.py_game.set_timer(USEREVENT + 2, 500)

        print('[CONTROLLER-THREAD]: init finished')
        controller_initialized.set()

    def run(self):
        print('[CONTROLLER-THREAD]: run started')

        global lz_finder, img_obj_det, exit_program, prev_error, img_obj_det_q, landing_zone_xy_q, frame_q  # , frame_global

        # frame_global = None
        prev_error = (0, 0, 0, 0)
        frame = self.drone.get_frame()
        img_april_tag = None
        img_obj_det_before = dummy_img_for_init

        while not exit_program:
            controller_loop_start_time = time.time()
            print('[CONTROLLER-THREAD]: run LOOP started')
            print(f'[CONTROLLER-THREAD]: {threading.active_count()}')
            flight_phase = self.drone.flight_phase

            rc_values = (0, 0, 0, 0)
            area = 0
            target_xy = (0, 0)
            landing_zone_xy = (0, 0)
            img_obj_det_new = None

            frame = self.drone.get_frame()
            print('[CONTROLLER-THREAD]: frame received from drone')

            for event in pygame.event.get():
                if event.type == USEREVENT + 2:
                    # frame_q.append(frame)
                    with frame_q.mutex:
                        frame_q.queue.clear()
                    frame_q.put(frame)
                    print('[CONTROLLER-THREAD]: frame put in queue')

                '''if event.type == USEREVENT + 1:
                    rc_values = tp.keyboard_rc(self.drone, rc_values, self.py_game, self.drone.speed)
                    self.drone.me.send_rc_control(rc_values[0], rc_values[1], rc_values[2], rc_values[3])'''
            print('[CONTROLLER-THREAD]: events pygame events handled')

            # register ESC has high priority
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
                if self.drone.flight_phase == ("Approach" or "Landing"):
                    self.auto_pilot.auto_pilot_armed = not self.auto_pilot.auto_pilot_armed

            # set flight phase
            if tp.takeoff_phase_key_pressed(self.py_game):
                if self.drone.me.is_flying is not True:
                    self.drone.me.takeoff()
                    self.drone.flight_phase = "Take-off"
            elif tp.hover_phase_key_pressed(self.py_game):
                self.drone.flight_phase = "Hover"
                if self.auto_pilot.auto_pilot_armed:
                    self.auto_pilot.auto_pilot_armed = False
            elif tp.approach_phase_key_pressed(self.py_game):
                self.drone.flight_phase = "Approach"
                print('[CONTROLLER-THREAD]: approach phase +++++++++++++++++')
            elif tp.landing_phase_key_pressed(self.py_game):
                self.drone.flight_phase = "Landing"

            # auto_pilot computations
            if self.drone.flight_phase == "Take-off":
                pass

            elif self.drone.flight_phase == "Hover":
                pass

            elif self.drone.flight_phase == "Approach":
                if self.auto_pilot.auto_pilot_armed:
                    img_april_tag, target_xy, area = self.apriltag_finder.apriltag_center_area(frame)
                else:
                    img_april_tag, _, _ = self.apriltag_finder.apriltag_center_area(frame)


            elif self.drone.flight_phase == "Landing":
                try:
                    img_obj_det_new = img_obj_det_q.get_nowait()
                except queue.Empty:
                    print('[CONTROLLER-THREAD]: queue is empty')

                # if self.auto_pilot.auto_pilot_armed:
                try:
                    landing_zone_xy = landing_zone_xy_q.get_nowait()
                except queue.Empty:
                    print('[CONTROLLER-THREAD]: landing_zone_xy_queue is empty')

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

            print('[CONTROLLER-THREAD]: landing_zone_xy: ' + str(landing_zone_xy))

            if target_xy is not None:
                # send rc commands based on target_xy
                error = self.auto_pilot.get_alignment_error(target_xy, area)
                rc_values = self.auto_pilot.track_target(rc_values, target_xy, error, prev_error, area)
                prev_error = error

            rc_values = tp.keyboard_rc(self.drone, rc_values, self.py_game, self.drone.speed)

            # send rc commands to drone; order: 1) keyboard, 2) auto_pilot, 3) default (0,0,0,0)
            self.drone.me.send_rc_control(rc_values[0], rc_values[1], rc_values[2], rc_values[3])

            # get drone sensor data for display in pygame
            battery_level, temperature, flight_time, _, _ = self.drone.get_drone_sensor_data()

            # video feed via pygame
            if self.drone.flight_phase == "Take-off":
                pass  # no video feed first, first cool down using fans
            elif self.drone.flight_phase == "Hover":
                # TESTING OUT SEG LIVE VIEW
                frame = lz_finder.segEngine.infer_image(frame)
                frame = ut.add_central_dot(frame)
                self.py_game.display_video_feed(self.screen, frame)
            elif self.drone.flight_phase == "Approach":
                img_april_tag = ut.add_central_dot(img_april_tag)
                self.py_game.display_video_feed(self.screen, img_april_tag)

            if self.drone.flight_phase == "Landing":
                frame = ut.add_central_dot(frame)
                self.py_game.display_video_feed(self.screen, frame)

                if img_obj_det_new is not None:
                    img_obj_det_new = cv.cvtColor(img_obj_det_new, cv.COLOR_BGR2RGB)
                    cv.imshow('Object Detection', img_obj_det_new)

            if battery_level <= 15:
                self.py_game.display_status(screen=self.screen, text="Battery level low: " + str(battery_level) + "%",
                                            show_warning=True)
            else:
                self.py_game.display_multiple_status(screen=self.screen, v_pos=10, h_pos=10,
                                                     battery_level=battery_level, flight_time=flight_time,
                                                     temperature=temperature, speed=self.drone.speed,
                                                     flight_phase=self.drone.flight_phase,
                                                     auto_pilot_armed=self.auto_pilot.auto_pilot_armed)
            pygame.display.flip()

            controller_loop_end_time = time.time()
            print("[CONTROLLER-THREAD]: Loop Time: " + str(
                (controller_loop_end_time - controller_loop_start_time) * 1000))

            time.sleep(1 / FPS)

        if exit_program:
            sys.exit()


def main():
    thread_yolo = threading.Thread(target=yolo_thread, args=())
    thread_yolo.daemon = True
    thread_yolo.start()

    yolo_drone = DroneController()
    yolo_drone.run()


if __name__ == '__main__':
    main()
