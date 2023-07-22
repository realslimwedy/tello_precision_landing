from ultralytics import YOLO
import cv2

# create webcam feed
cap = cv2.VideoCapture(0)
model = YOLO('./object_tracking/yolo_models/yolov8n-seg.pt')

while True:
    # read frame from webcam
    ret, frame = cap.read()
    yolo_results = model(frame)
    annotated_yolo_frame = yolo_results[0].plot()
    cv2.imshow("YOLOv8 Inference", annotated_yolo_frame)

    # press q to quit
    if cv2.waitKey(1) == ord('q'):
        break

