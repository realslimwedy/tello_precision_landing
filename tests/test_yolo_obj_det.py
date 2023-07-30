import code.vision_package as vp
import code.utils as ut
import time
import cv2 as cv

labels_str_list_blacklist = ['train', 'stop sign', 'bottle', 'carrot', "dining table"]
labels_str_list_whitelist = [cls for cls in vp.labels.labels_yolo.keys() if cls not in labels_str_list_blacklist]
labels_dic_filtered = {key: value for key, value in vp.labels.labels_yolo.items() if key in labels_str_list_whitelist}

model_path = '../code/yoloV8_models/yolov8n.pt'


def test_yolo_obj_det():
    start_time_model_load = time.time()
    object_detector = vp.ObjectDetector(model_path, labels_dic_filtered, verbose=True, max_det=None)
    end_time_model_load = time.time()
    ut.print_interval_ms('Model Load Time', start_time_model_load, end_time_model_load)

    cap = cv.VideoCapture(0)

    list_of_loop_times = []

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.resize(frame, (640, 480))

        height, width = frame.shape[:2]

        img, obstacles = object_detector.infer_image(height, width, frame)

        cv.imshow("YOLOv8 Object Detection TEST", img)
        cv.waitKey(1)

        end_time = time.time()

        avg_value, list_of_loop_times = ut.rolling_average_of_float_values(list_of_loop_times, end_time - start_time, 5)

        print('Yolo Object Detection Test (isolated process)')
        ut.print_interval_ms('Loop Time', start_time, end_time)
        ut.print_time_ms("Loop Time Avg", avg_value)
        ut.print_fps("FPS", start_time, end_time)

        if cv.waitKey(1) == ord('q'):
            break

        key = cv.waitKey(1)
        if key == 27:
            break


if __name__ == '__main__':
    test_yolo_obj_det()
