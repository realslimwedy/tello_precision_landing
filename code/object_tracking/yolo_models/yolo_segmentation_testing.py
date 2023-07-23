import sys

from ultralytics import YOLO
import cv2

# Create webcam feed
cap = cv2.VideoCapture(0)
model = YOLO('./object_tracking/yolo_models/yolov8n-seg.pt')

# Load the class names from the YOLO model
class_names = model.names


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Perform object detection and segmentation on the frame
    results = model(frame)
    print(model.names)
    masks = results[0].masks
    boxes = results[0].boxes

    annotated_yolo_frame = results[0].plot()
    cv2.imshow("YOLOv8 Inference", annotated_yolo_frame)



    # Now you can extract and print the data from masks
    print(masks.xy)  # x, y segments (pixels), List[segment] * N
    print()
    print(masks.xyn)  # x, y segments (normalized), List[segment] * N
    print()
    print(masks.data)
    print()
    print(boxes.cls)


    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
