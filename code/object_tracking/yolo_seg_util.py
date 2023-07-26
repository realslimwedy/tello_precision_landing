import cv2 as cv
from ultralytics import YOLO
import numpy as np


class SegmentationEngine():
    def __init__(self, model, label_ids):
        self.model = model
        self.label_ids = label_ids

    def __str__(self):
        return f"SegmentationEngine instance with model: {self.model}"

    def inferImage(self, img):
        width, height = img.shape[1], img.shape[0]

        self.model.predict(source=img, verbose=False, classes=self.label_ids)

        results = self.model(img)
        boxes = results[0].boxes
        masks = results[0].masks

        output_array = np.full((height, width), -1, dtype=int)
        # '-1' for unlabelled pixels because '0' implies class 'person' in YOLOv8

        for i in range(len(boxes)):
            cls = int(boxes[i].cls.tolist()[0])
            mask_data = masks.data[i].tolist()

            # Get the height and width from the shape of the masks tensor
            mask_height, mask_width = masks.shape[1], masks.shape[2]

            # Calculate scaling factors
            scale_x = width / mask_width
            scale_y = height / mask_height

            for y, row in enumerate(mask_data):
                for x, val in enumerate(row):
                    if val == 1:
                        # Scale the coordinates back to the original image resolution
                        scaled_x = int(x * scale_x)
                        scaled_y = int(y * scale_y)
                        if scaled_x < width and scaled_y < height:
                            output_array[scaled_y, scaled_x] = cls

        output_array = np.array(output_array, dtype=np.uint8)

        return output_array

    def inferImageDummy(self, img):
        width, height = img.shape[1], img.shape[0]
        output_array = np.full((height, width), 255, dtype=np.uint8)
        return output_array


if __name__ == "__main__":
    label_list = ["apple", "banana", "background", "book", "person"]
    labels = {key: value for key, value in labelsYolo.items() if key in label_list}
    label_ids = list(labels.values())

    model = YOLO('yoloV8_models/yolov8n-seg.pt')
    seg_engine = SegmentationEngine(model, label_ids)

    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to retrieve a frame from the camera.")
            break
        frame = cv.resize(frame, (320, 240))  # 640, 480 vs. 320, 240

        segImg = seg_engine.inferImage(frame)

        cv.imshow('frame', segImg)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
