from ultralytics import YOLO
import cv2 as cv
import time


class ObjectDetector():
    def __init__(self, model_path, labels_dic_filtered, max_det=None):
        self.model = YOLO(model_path)
        self.labels_dic_filtered = labels_dic_filtered
        self.max_det = max_det

        self.labels_ids_list_whitelist = list(labels_dic_filtered.values())

        self.labels_dic_filtered_inverted = {value: key for key, value in labels_dic_filtered.items()}
        self.color = [0, 255, 0]  # Green

    def __str__(self):
        return f"ObjectDetector instance with model: {self.model}"

    def draw_labels_and_boxes(self, img, boxes, confidences, classids):
        # If there are any detections
        if len(boxes) > 0:
            for i in range(len(boxes)):
                # Get the bounding box coordinates
                x, y = int(boxes[i][0]), int(boxes[i][1])
                w, h = int(boxes[i][2]), int(boxes[i][3])

                # Draw the bounding box rectangle and label on the image
                cv.rectangle(img, (x, y), (x + w, y + h), self.color, 2)

                text = "{}: {:4f}".format(self.labels_dic_filtered_inverted[classids[i]], confidences[i])

                cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 2)

        return img

    def generate_boxes_confidences_classids(self, objects, height, width, tconf):

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
                if cls in self.labels_dic_filtered_inverted:
                    x = int(xywh[0] - (xywh[2] / 2))
                    y = int(xywh[1] - (xywh[3] / 2))
                    boxes.append([x, y, xywh[2], xywh[3]])
                    confidences.append(float(conf))
                    classids.append(cls)

        return boxes, confidences, classids

    def infer_image(self, height, width, img, boxes=None, confidences=None, classids=None, infer=True, confidence=0.25,
                    drawBoxes=True, ):
        '''
        Function returns the labelled image and the obstacles detected
        obstacles are returned as a list of dictionaries
        an obstacle consists of an id, confidence, and bounding box
        '''

        if infer:
            results = self.model.predict(img, verbose=True, classes=self.labels_ids_list_whitelist,
                                         max_det=self.max_det)  # this takes 99% of the time

            objects = results[0].boxes
            # Generate the boxes, confidences, and classIDs
            boxes, confidences, classids = self.generate_boxes_confidences_classids(objects, height, width, confidence)

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

    import labels

    labels_str_list_blacklist = ['train', 'stop sign', 'bottle', 'carrot', "dining table"]
    labels_str_list_whitelist = [cls for cls in labels.labels_yolo.keys() if cls not in labels_str_list_blacklist]
    labels_dic_filtered = {key: value for key, value in labels.labels_yolo.items() if key in labels_str_list_whitelist}

    model_path = 'yoloV8_models/yolov8n.pt'

    object_detector = ObjectDetector(model_path, labels_dic_filtered)
    cap = cv.VideoCapture(0)

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.resize(frame, (640, 480))

        height, width = frame.shape[:2]

        img, obstacles = object_detector.infer_image(height, width, frame)

        cv.imshow("YOLOv8 Inference", img)
        cv.waitKey(1)

        stop_time = time.time()
        print(f'Loop Time: {(stop_time - start_time) * 1000} ms')
        print(f"FPS: {1 / (stop_time - start_time)}")

        if cv.waitKey(1) == ord('q'):
            break
