import cv2 as cv
from ultralytics import YOLO
import numpy as np
import time


class SegmentationEngine():
    def __init__(self, model_path, labels_dic_filtered):
        self.model = YOLO(model_path)
        self.labels_ids_list_filtered = list(labels_dic_filtered.values())

        print(self.labels_ids_list_filtered)
        self.verbose = False

        self.background_index = 80  # must be one greater than the last index in labels_yolo  # this is a workaround:  # main reason: a person is class 0 in YOLOv8, so background needs to be indexed differently  # using -1 would yield 255 during conversion into uint8 in inferImage()  # workaround: use 80, which is the index of the last class in labels_yolo +1  # self.background_index must a unique value and stated in labels.risk_labels  # for later conversion into risk_map

    def __str__(self):
        return f"SegmentationEngine instance with model: {self.model}"

    def infer_image(self, img):
        width, height = img.shape[1], img.shape[0]

        results = self.model.predict(source=img, verbose=self.verbose,
                                     classes=self.labels_ids_list_filtered)  # this list
        boxes = results[0].boxes  # <class 'ultralytics.engine.results.Boxes'>
        masks = results[0].masks  # <class 'ultralytics.engine.results.Masks'>

        output_array_with_class_predictions = np.full((height, width), self.background_index, dtype=int)

        for i in range(len(boxes)):
            cls_i = int(boxes[i].cls.tolist()[0])  # type int
            mask_list = masks.data[i].tolist()  # type list with 0s and 1s indicating where class is present

            # Get the height and width from the shape of the masks tensor
            mask_height, mask_width = masks.shape[1], masks.shape[2]  # type int, corresponds to height, width of img

            # Calculate scaling factors
            scale_x = width / mask_width
            scale_y = height / mask_height

            for y, row in enumerate(mask_list):
                for x, val in enumerate(row):
                    if val == 1:
                        # Scale the coordinates back to the original image resolution
                        scaled_x = int(x * scale_x)
                        scaled_y = int(y * scale_y)
                        if scaled_x < width and scaled_y < height:
                            output_array_with_class_predictions[scaled_y, scaled_x] = cls_i

        output_array_with_class_predictions = np.array(output_array_with_class_predictions, dtype=np.uint8)

        return output_array_with_class_predictions

    def infer_image_dummy(self, img):
        width, height = img.shape[1], img.shape[0]
        output_array = np.full((height, width), self.background_index, dtype=np.uint8)
        return output_array


if __name__ == "__main__":
    import labels

    labels_str_list_blacklist = ['train', 'stop sign', 'bottle', 'carrot', "dining table"]
    print(labels_str_list_blacklist)
    labels_str_list_whitelist = [cls for cls in labels.labels_yolo.keys() if cls not in labels_str_list_blacklist]
    print(labels_str_list_whitelist)
    labels_dic_filtered = {key: value for key, value in labels.labels_yolo.items() if key in labels_str_list_whitelist}
    print(labels_dic_filtered)
    labels_ids_list_filtered = list(labels_dic_filtered.values())

    model_path = 'yoloV8_models/yolov8n-seg.pt'

    seg_engine = SegmentationEngine(model_path, labels_dic_filtered=labels_dic_filtered)

    cap = cv.VideoCapture(0)

    while True:
        start_time = time.time()

        # Frame Read
        start_time_frame_read = time.time()
        ret, frame = cap.read()
        end_time_frame_read = time.time()
        print(f'Frame Read Time: {round((end_time_frame_read - start_time_frame_read) * 1000, 1)} ms')

        if not ret:
            print("Error: Failed to retrieve a frame from the camera.")
            break

        # Frame Resize
        start_time_frame_resize = time.time()
        frame = cv.resize(frame, (640, 480))  # 320,240
        end_time_frame_resize = time.time()
        print(f'Frame Resize Time: {round((end_time_frame_resize - start_time_frame_read) * 1000, 1)} ms')

        # Inference
        start_time_inference = time.time()
        # 1) raw seg image, normal colors & annotations
        '''results = seg_engine.model.predict(source=frame, verbose=True, classes=labels_ids_list_filtered)
        annotated_frame = results[0].plot()
        cv.imshow('YOLOv8 Inference', annotated_frame)'''

        # 2) grayscale colors and scaled back to frame resolution
        seg_engine.labels_str_list_filtered = None
        seg_engine.verbose = True
        segImg = seg_engine.infer_image(frame)
        cv.imshow('frame', segImg)

        end_time_inference = time.time()
        print(f'Inference Time: {round((end_time_inference - start_time) * 1000, 1)} ms')

        end_time = time.time()
        print(f'Loop Time: {round((end_time - start_time) * 1000, 1)} ms')
        print(f"FPS: {round(1 / (end_time - start_time))}")

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
