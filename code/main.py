import pygame, cv2, time, sys, asyncio
from ultralytics import YOLO
from djitellopy import tello
import tello_package as tp
import object_tracking as ot


if __name__ == '__main__':

    model_obj_det = YOLO('./object_tracking/yolo_models/yolov8n.pt')
    print("Object Detection Model loaded")
    model_seg = YOLO('./object_tracking/yolo_models/yolov8n-seg.pt')
    print("Segmentation Model loaded")

    label_list = ["apple", "banana", "background", "book", "person"]
    lzFinder = ot.LzFinder(model_obj_det=model_obj_det, model_seg=model_seg,res=(640, 480), label_list=label_list, use_seg=True, r_landing_factor=8,
                        stride=75)  # 640, 480 vs. 320, 240
    print(lzFinder)

    model_obj_det = YOLO('./object_tracking/yolo_models/yolov8n.pt')
    model_seg = YOLO('./object_tracking/yolo_models/yolov8n-seg.pt')

    #objectDetector = ot.ObjectDetector(model_obj_det)

    width, height = 640, 480
    res = (width, height)
    exit_program = False

    # init pygame
    py_g = tp.Pygame()
    screen = pygame.display.set_mode(res)
    battery_font = pygame.font.SysFont(None, 25)

    # init drone
    drone=tp.Drone(res=res, mirror_down=True)
    drone.power_up()
    rc_params = (0, 0, 0, 0)
    img = drone.get_frame()

    # init auto_pilot
    auto_pilot =  tp.Autopilot(res=res,speed=25,apriltag_factor=1)
    autopilot_armed = auto_pilot.autopilot_armed
    prv_error = (0, 0, 0, 0)

    async def main():

        async def show_video_feed():
            global img, autopilot_armed, model_obj_det, height, width
            while True:
                img = drone.get_frame()
                autopilot_armed = auto_pilot.autopilot_armed

                await asyncio.sleep(1. / 30)

                # video feed via cv2
                '''img_cv2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cv2.imshow('Video Feed', img_cv2)
                cv2.waitKey(1)'''

                #img_yolo = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                #img_yolo, obstacles = objectDetector.infer_image(height, width, img)

                '''yolo_results = model(img_yolo)
                annotated_yolo_frame = yolo_results[0].plot()
                cv2.imshow("YOLOv8 Inference", annotated_yolo_frame)'''

                # video feed via pygame
                battery_level, temperature, flight_time, barometer, distance_tof = drone.get_drone_sensor_data()
                #alt, speed_x, speed_y, speed_z, acc_x, acc_y, acc_z, roll, pitch, yaw = drone.get_flight_state()

                auto_pilot_text = "Auto Pilot: On" if auto_pilot.autopilot_armed else "Auto Pilot: Off"
                flight_phase_text = f"Flight Phase: {drone.flight_phase}"

                py_g.display_video_feed(screen, img)
                py_g.display_status(screen=screen, text=f'Battery Level: {battery_level} %', v_position=10, h_position=10)
                #py_g.display_status(screen=screen, text=f'Temperature: {temperature} °C', v_position=10,h_position=40)
                #py_g.display_status(screen=screen, text=f'Flight Time: {flight_time} Seconds', v_position=10,h_position=70)
                py_g.display_status(screen=screen, text=auto_pilot_text,   v_position=10, h_position=100)
                py_g.display_status(screen=screen, text=flight_phase_text, v_position=10, h_position=130)
                #py_g.display_status(screen=screen, text=f'Altitude: {alt}', v_position=10, h_position=160)
                #py_g.display_status(screen=screen, text=f'Speed_X: {speed_x}',  v_position=10, h_position=190)
                #py_g.display_status(screen=screen, text=f'Speed_Y: {speed_y}',  v_position=10, h_position=220)
                #py_g.display_status(screen=screen, text=f'Speed_Z: {speed_z}', v_position=10, h_position=250)

                pygame.display.flip()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

                if tp.exit_app_key_pressed(drone.me, py_g):
                    drone.power_down()
                    cv2.destroyAllWindows()
                    pygame.quit()
                    exit_program = True
                    break

            if exit_program == True:
                sys.exit()

        async def drone_control():
            global img, auto_pilot, prv_error
            while True:
                rc_params = (0, 0, 0, 0)
                target_xy = None
                area = 0

                if tp.autopilot_key_pressed(drone.me, py_g):
                    if not auto_pilot.autopilot_armed:
                        auto_pilot.arm_autopilot()
                    elif auto_pilot.autopilot_armed:
                        auto_pilot.arm_autopilot()

                if tp.takeoff_phase_key_pressed(drone.me, py_g):
                    drone.flight_phase = "Take-off"
                elif tp.approach_phase_key_pressed(drone.me, py_g):
                    drone.flight_phase = "Approach"
                elif tp.landing_phase_key_pressed(drone.me, py_g):
                    drone.flight_phase = "Landing"

                # if auto_pilot.autopilot_armed:
                if drone.flight_phase == "Take-off":
                    pass
                # drone.me.send_rc_control(0, 0, 0, auto_pilot.speed)
                elif drone.flight_phase == "Approach":
                    img, target_xy, area = ot.apriltag_center_area(img)

                elif drone.flight_phase == "Landing":
                    landing_zone_xy, img, risk_map = lzFinder.get_final_lz(img)
                    area = None

                if target_xy is not None:
                    error = auto_pilot.get_alignment_error(target_xy, area)
                    rc_params = auto_pilot.track_target(rc_params, target_xy,error, prv_error, area)
                    prv_error = error
                else:
                    print("Center is None. Skipping auto-pilot processing.")


                rc_params = tp.keyboard_rc(drone.me, rc_params, py_g)

                drone.me.send_rc_control(rc_params[0], rc_params[1], rc_params[2], rc_params[3])

                await asyncio.sleep(0.01)

        await asyncio.gather(show_video_feed(), drone_control())


    asyncio.run(main())
