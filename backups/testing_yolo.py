from ultralytics import YOLO
import cv2 as cv
import time
from labels import labels_yolo

if __name__ == '__main__':
    label_list = ["apple", "banana", "background", "book", "person"]
    labels = {key: value for key, value in labels_yolo.items() if key in label_list}
    label_ids = list(labels.values())

    model = YOLO('../code/vision_package/yoloV8_models/yolov8n.pt')

    cap = cv.VideoCapture(0)

    while True:

        ret, frame = cap.read()
        frame = cv.resize(frame, (640, 480))

        start_time = time.time()  # Record the start time of the iteration
        results = model(frame)
        end_time = time.time()  # Record the end time of the iteration

        print(f"Time taken for iteration: {end_time - start_time:.5f} seconds")

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv.imshow("YOLOv8 Inference", annotated_frame)

        if not ret:
            break


        if cv.waitKey(1) == ord('q'):
            break