import cv2 as cv
from ultralytics import YOLO
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

class SegmentationEngine():
    def __init__(self, model):
        self.model = model

    def inferImage(self, img):
        width, height = img.shape[1], img.shape[0]
        results = self.model(img)
        boxes = results[0].boxes
        masks = results[0].masks

        output_array = np.zeros((height, width), dtype=int)

        for i in range(len(boxes)):
            cls = int(boxes[i].cls.tolist()[0])
            mask_data = masks.data[i].tolist()

            for y, row in enumerate(mask_data):
                for x, val in enumerate(row):
                    if val == 1:
                        output_array[y,x] = cls

        output_image = Image.fromarray(output_array.astype('uint8'))

        return output_image


if __name__ == "__main__":
    model = model = YOLO('./object_tracking/yolo_models/yolov8n-seg.pt')
    segEngine = SegmentationEngine(model)

    cap = cv.VideoCapture(0)

    '''ret, frame = cap.read()
    frame = cv.resize(frame, (640, 480))
    pred_mask = segEngine.inferImage(frame)
    pred_mask_np = np.array(pred_mask)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(frame[:, :, ::-1])
    ax1.set_title("Picture")
    ax2.imshow(pred_mask_np, cmap='gray')
    ax2.set_title("Prediction")
    ax2.set_axis_off()
    plt.show()'''

    while True:
        ret, frame = cap.read()
        frame = cv.resize(frame, (640, 480))
        pred_mask = segEngine.inferImage(frame)

        # Find the maximum value in pred_mask
        max_value = np.amax(pred_mask)

        # Normalize pred_mask to a scale of 0 to 255
        pred_mask_normalized = (pred_mask / max_value) * 255

        # Convert pred_mask_normalized to an 8-bit unsigned integer array
        pred_mask_normalized = pred_mask_normalized.astype(np.uint8)

        # Show pred continuously in a window
        cv.imshow('frame', pred_mask_normalized)

        # Press q to quit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break








