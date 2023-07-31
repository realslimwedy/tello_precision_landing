import code.vision_package as vp
import code.utils as ut
import time
import cv2 as cv

labels_str_list_blacklist = ['train', 'stop sign', 'bottle', 'carrot', "dining table"]
labels_str_list_whitelist = [cls for cls in vp.labels.labels_yolo.keys() if cls not in labels_str_list_blacklist]
labels_dic_filtered = {key: value for key, value in vp.labels.labels_yolo.items() if key in labels_str_list_whitelist}

labels_ids_list_filtered = list(labels_dic_filtered.values())

model_path = '../code/yoloV8_models/yolov8n-seg.pt'

def test_yolo_obj_det_raw_image():
    start_time_model_load = time.time()
    seg_engine = vp.SegmentationEngine(model_path, labels_dic_filtered, verbose=True)
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

        results = seg_engine.model.predict(source=frame, verbose = True, classes = labels_ids_list_filtered)
        annotated_frame = results[0].plot()
        #cv.imshow('Yolo Segmentation Test - Raw Yolo Output (isolated process)', annotated_frame)
        cv.waitKey(1)

        end_time = time.time()

        avg_value, list_of_loop_times = ut.rolling_average_of_float_values(list_of_loop_times, end_time - start_time, 5)

        print('Yolo Segmentation Test - Raw Yolo Output (isolated process)')
        ut.print_interval_ms('Loop Time', start_time, end_time)
        ut.print_time_ms("Loop Time Avg", avg_value)
        ut.print_fps("FPS", start_time, end_time)

        if cv.waitKey(1) == ord('q'):
            break

        key = cv.waitKey(1)
        if key == 27:
            break

def test_yolo_obj_det_infer_image():
    start_time_model_load = time.time()
    seg_engine = vp.SegmentationEngine(model_path, labels_dic_filtered, verbose=True)
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

        output_array_with_class_predictions = seg_engine.infer_image(height, width, frame)

        cv.imshow("Yolo Segmentation Test - Infer Image (isolated process)", output_array_with_class_predictions)
        cv.waitKey(1)

        end_time = time.time()

        avg_value, list_of_loop_times = ut.rolling_average_of_float_values(list_of_loop_times, end_time - start_time, 5)

        print('Yolo Segmentation Test - Infer Image (isolated process)')
        ut.print_interval_ms('Loop Time', start_time, end_time)
        ut.print_time_ms("Loop Time Avg", avg_value)
        ut.print_fps("FPS", start_time, end_time)

        if cv.waitKey(1) == ord('q'):
            break

        key = cv.waitKey(1)
        if key == 27:
            break



if __name__ == '__main__':
    #test_yolo_obj_det_raw_image()
    test_yolo_obj_det_infer_image()

