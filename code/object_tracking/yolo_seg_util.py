import cv2 as cv
from ultralytics import YOLO
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

class SegmentationEngine():
    def __init__(self, model):
        self.model = model

    def __str__(self):
        return f"SegmentationEngine instance with model: {self.model}"

    def inferImage(self, img):
        width, height = img.shape[1], img.shape[0]
        self.model.predict(img, classes=[0,46,47,73],verbose=False)
        results = self.model(img)
        boxes = results[0].boxes
        masks = results[0].masks

        output_array = np.full((height, width),-1, dtype=int)
        # '-1' for unlabelled pixels because '0' implies class 'person' in YOLOv8

        for i in range(len(boxes)):
            cls = int(boxes[i].cls.tolist()[0])
            mask_data = masks.data[i].tolist()

            for y, row in enumerate(mask_data):
                for x, val in enumerate(row):
                    if val == 1:
                        output_array[y,x] = cls

        # save output_array as txt file
        np.savetxt('pred_mask.txt', output_array.reshape(-1), delimiter=',', fmt='%d')

        output_array = np.array(output_array, dtype=np.uint8)
        # save output_array as txt file
        np.savetxt('pred_mask_II.txt', output_array.reshape(-1), delimiter=',', fmt='%d')

        return output_array


if __name__ == "__main__":
    model = YOLO('./object_tracking/yolo_models/yolov8n-seg.pt')
    segEngine = SegmentationEngine(model)

    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        cap.release()
        cv.destroyAllWindows()
        exit()

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
        if not ret:
            print("Error: Failed to retrieve a frame from the camera.")
            break
        frame = cv.resize(frame, (640, 480))

        segImg = segEngine.inferImage(frame)

        cv.imshow('frame', segImg)

        # Press q to quit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break








