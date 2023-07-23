import sys

from ultralytics import YOLO
from ultralytics.utils import ops
import cv2

# create webcam feed
cap = cv2.VideoCapture(0)
model = YOLO('./object_tracking/yolo_models/yolov8n.pt')

ret, frame = cap.read()
results = model(frame)
boxes = results[0].boxes.data
boxes_nms=ops.non_max_suppression(boxes)
print(boxes_nms)

sys.exit()