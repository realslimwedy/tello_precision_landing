import sys
import numpy as np
from ultralytics import YOLO
import cv2

# Create webcam feed
cap = cv2.VideoCapture(0)
model = YOLO('./object_tracking/yolo_models/yolov8n-seg.pt')

for i in range(1):
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)

    masks = results[0].masks
    boxes = results[0].boxes
    print('###############################################')

    #print('masks.data')
    #print(masks.data)
    print()
    print('boxes.cls')
    print(boxes.cls)
    print()
    print('masks.data')
    print(masks.data)  # x, y segments (pixels), List[segment] * N
    print()


    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
