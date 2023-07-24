from ultralytics import YOLO
import cv2 as cv
import numpy as np

class ObjectDetector():
    def __init__(self,model_obj_det):
        self.labels={0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
        self.model_obj_det = model_obj_det
        self.color = [0, 255, 0]

    def __str__(self):
        return f"ObjectDetector instance with model: {self.model_obj_det}"

    def show_image(self,img):
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
                text = "{}: {:4f}".format(self.labels[classids[i]], confidences[i])
                cv.putText(
                    img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 2
                )
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

        #print()
        #print('FOR LOOP')
        for obj in objects:
            #print(obj.xywh)
            xywh = obj.xywh.tolist()[0]
            #print(xywh)
            #print()

            #print(obj.conf)
            conf = obj.conf.tolist()[0]
            #print(conf)
            #print()

            #print(obj.cls)
            cls = int(obj.cls.tolist()[0])
            #print(cls)
            #print()
            if conf > tconf:
                x = int(xywh[0] - (xywh[2] / 2))
                y = int(xywh[1] - (xywh[3] / 2))
                boxes.append([x,y, xywh[2], xywh[3]])
                confidences.append(float(conf))
                classids.append(cls)

        return boxes, confidences, classids


    def infer_image(
            self,
            height,
            width,
            img,
            boxes=None,
            confidences=None,
            classids=None,
            idxs=None,
            infer=True,
            confidence=0.25,
            threshold=0.3,
            drawBoxes=True,
    ):
        '''
        Function returns the labelled image and the obstacles detected
        obstacles are returned as a list of dictionaries
        an obstacle consists of an id, confidence, and bounding box
        '''

        if infer:
            self.model_obj_det.predict(img, classes=[0,46,47,73],verbose=False)
            results= self.model_obj_det(img)

            # Generate the boxes, confidences, and classIDs
            boxes, confidences, classids = self.generate_boxes_confidences_classids(
                results, height, width, confidence
            )

            # Apply Non-Maxima Suppression to suppress overlapping bounding boxes
            #idxs = cv.dnn.NMSBoxes(boxes, confidences, confidence, threshold)

        if boxes is None or confidences is None or classids is None:
            raise "[ERROR] Required variables are set to None before drawing boxes on images."

        obstacles = []
        if len(boxes) > 0:
            for i in range(len(boxes)):
                obstDetected = {
                    "label": classids[i],
                    "confidence": confidences[i],
                    "box": boxes[i],
                }
                obstacles.append(obstDetected)
        # Draw labels and boxes on the image
        if drawBoxes:
            img = self.draw_labels_and_boxes(img, boxes, confidences, classids)
        return img, obstacles


if __name__ == '__main__':
    model_obj_det = YOLO('./object_tracking/yolo_models/yolov8n.pt')
    objectDetector = ObjectDetector(model_obj_det)
    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        height, width = frame.shape[:2]

        img, obstacles = objectDetector.infer_image(height, width, frame)
        objectDetector.show_image(img)
        # add sleep of 1 second
        #time.sleep(1)
        #annotated_yolo_frame = results[0].plot()

        # press q to quit
        if cv.waitKey(1) == ord('q'):
            break