import code.vision_package as vp
import code.utils as ut
import time
import cv2 as cv

labels_str_list_blacklist = ['train', 'stop sign', 'bottle', 'carrot', "dining table"]
labels_str_list_whitelist = [cls for cls in vp.labels.labels_yolo.keys() if cls not in labels_str_list_blacklist]
labels_dic_filtered = {key: value for key, value in vp.labels.labels_yolo.items() if key in labels_str_list_whitelist}

labels_ids_list_filtered = list(labels_dic_filtered.values())

model_obj_det_path = '../code/yoloV8_models/yolov8n.pt'
model_seg_path = '../code/yoloV8_models/yolov8n-seg.pt'

RES = (640, 480)
HEIGHT, WIDTH = RES


def test_landing_zone_finder(output='img', imshow=True):
    start_time_model_load = time.time()

    lz_finder = vp.LzFinder(model_obj_det_path=model_obj_det_path, model_seg_path=model_seg_path,
                            labels_dic_filtered=labels_dic_filtered, max_det=None, res=RES, use_seg_for_lz=False,
                            r_landing_factor=8, stride=75, verbose=True)

    end_time_model_load = time.time()
    ut.print_interval_ms('LZ Finder Load Time', start_time_model_load, end_time_model_load)

    cap = cv.VideoCapture(0)

    list_of_lz_tuples = []
    list_of_loop_times = []

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.resize(frame, RES)

        landing_zone_xy, img, risk_map = lz_finder.get_final_lz(frame)

        if output == 'risk_map':
            risk_map = cv.applyColorMap(risk_map, cv.COLORMAP_BONE)
            if imshow:
                cv.imshow("Landing Zone Finder Test - RISK MAP (isolated process)", risk_map)
        else:
            if imshow:
                cv.imshow("Landing Zone Finder Test - IMG (isolated process)", img)

        cv.waitKey(1)

        end_time = time.time()

        landing_zone_xy_avg, list_of_lz_tuples = ut.rolling_average_of_tuples(list_of_tuples=list_of_lz_tuples,
                                                                              new_tuple=landing_zone_xy,
                                                                              number_of_values=5)

        loop_time_avg, list_of_loop_times = ut.rolling_average_of_float_values(list_of_loop_times,
                                                                               end_time - start_time, 5)

        print(f'Landing Zone: {landing_zone_xy}')
        print(f'Landing Zone Avg: {landing_zone_xy_avg}')

        print('Yolo Segmentation Test - Infer Image (isolated process)')
        ut.print_interval_ms('Loop Time', start_time, end_time)
        ut.print_time_ms("Loop Time Avg", loop_time_avg)
        ut.print_fps("FPS", start_time, end_time)

        if cv.waitKey(1) == ord('q'):
            break

        key = cv.waitKey(1)
        if key == 27:
            break


if __name__ == '__main__':
    test_landing_zone_finder(output='img',
                             imshow=True)  # img vs. risk_map  # since lz_finder with seg=True is too slow > switched off  # hence 'risk_map' yields blank screen
