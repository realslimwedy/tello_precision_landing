from ultralytics import YOLO
import cv2 as cv
from .labels import labels_yolo


class ObjectDetector():
    def __init__(self, model, labelsYolo, label_ids):
        self.labels_inverted = {value: key for key, value in labelsYolo.items()}
        self.label_ids = label_ids
        self.model = model
        self.color = [0, 255, 0]

    def __str__(self):
        return f"ObjectDetector instance with model: {self.model}"

    def show_image(self, img):
        cv.imshow("YOLOv8 Inference", img)
        cv.waitKey(1)

    def draw_labels_and_boxes(self, img, boxes, confidences, classids):
        # If there are any detections
        if len(boxes) > 0:
            for i in range(len(boxes)):
                # Get the bounding box coordinates
                x, y = int(boxes[i][0]), int(boxes[i][1])
                w, h = int(boxes[i][2]), int(boxes[i][3])

                # Draw the bounding box rectangle and label on the image
                cv.rectangle(img, (x, y), (x + w, y + h), self.color, 2)

                text = "{}: {:4f}".format(self.labels_inverted[classids[i]], confidences[i])

                cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 2)

        return img

    def generate_boxes_confidences_classids(self, results, height, width, tconf):
        objects = results[0].boxes
        '''print('ENTIRE OBJECT')
        print(objects)
        print()
        print('FIRST OBJECT')
        print(objects[0])
        print()
        print()
        print('OBJECTS COMPONENTS')
        print(objects.xywh)
        print()
        print(objects.conf)
        print()
        print(objects.cls)'''

        boxes = []
        confidences = []
        classids = []

        # print()
        # print('FOR LOOP')
        for obj in objects:
            # print(obj.xywh)
            xywh = obj.xywh.tolist()[0]
            # print(xywh)
            # print()

            # print(obj.conf)
            conf = obj.conf.tolist()[0]
            # print(conf)
            # print()

            # print(obj.cls)
            cls = int(obj.cls.tolist()[0])
            # print(cls)
            # print()
            if conf > tconf:
                x = int(xywh[0] - (xywh[2] / 2))
                y = int(xywh[1] - (xywh[3] / 2))
                boxes.append([x, y, xywh[2], xywh[3]])
                confidences.append(float(conf))
                classids.append(cls)

        return boxes, confidences, classids

    def infer_image(self, height, width, img, boxes=None, confidences=None, classids=None, idxs=None, infer=True,
                    confidence=0.25, threshold=0.3, drawBoxes=True, ):
        '''
        Function returns the labelled image and the obstacles detected
        obstacles are returned as a list of dictionaries
        an obstacle consists of an id, confidence, and bounding box
        '''

        if infer:
            self.model.predict(img, verbose=False, classes=self.label_ids)  # classes=[0,46,47,73]
            results = self.model(img)

            # Generate the boxes, confidences, and classIDs
            boxes, confidences, classids = self.generate_boxes_confidences_classids(results, height, width, confidence)

            # Apply Non-Maxima Suppression to suppress overlapping bounding boxes  # idxs = cv.dnn.NMSBoxes(boxes, confidences, confidence, threshold)

        if boxes is None or confidences is None or classids is None:
            raise "[ERROR] Required variables are set to None before drawing boxes on images."

        obstacles = []
        if len(boxes) > 0:
            for i in range(len(boxes)):
                obstDetected = {"label": classids[i], "confidence": confidences[i], "box": boxes[i], }
                obstacles.append(obstDetected)

        if drawBoxes:
            img = self.draw_labels_and_boxes(img, boxes, confidences, classids)
        return img, obstacles


if __name__ == '__main__':
    label_list = ["apple", "banana", "background", "book", "person"]
    labels = {key: value for key, value in labels_yolo.items() if key in label_list}
    label_ids = list(labels.values())

    model = YOLO('yoloV8_models/yolov8n.pt')

    object_detector = ObjectDetector(model, labels, label_ids)
    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        frame = cv.resize(frame, (320, 240))

        if not ret:
            break

        height, width = frame.shape[:2]

        img, obstacles = object_detector.infer_image(height, width, frame)
        object_detector.show_image(img)

        if cv.waitKey(1) == ord('q'):
            break
