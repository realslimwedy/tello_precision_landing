from ultralytics import YOLO
import cv2 as cv


class ObjectDetector():
    def __init__(self, model_path, labels_dic_filtered, verbose=False, max_det=None):
        self.model = YOLO(model_path)
        self.labels_dic_filtered = labels_dic_filtered
        self.max_det = max_det
        self.verbose = verbose
        self.labels_ids_list_whitelist = list(labels_dic_filtered.values())

        self.labels_dic_filtered_inverted = {value: key for key, value in labels_dic_filtered.items()}
        self.color = [0, 0, 0]  # black

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

                text_x = x + 5
                text_y = y + 20

                text = "{}: {:4f}".format(self.labels_dic_filtered_inverted[classids[i]], confidences[i])

                cv.putText(img, text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 2)

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

    def infer_image(self, height, width, img, boxes=None, confidences=None, class_ids=None, infer=True, confidence=0.25,
                    draw_boxes=True, ):
        '''
        Function returns the labelled image and the obstacles detected
        obstacles are returned as a list of dictionaries
        an obstacle consists of an id, confidence, and bounding box
        '''

        if infer:
            results = self.model.predict(img, verbose=self.verbose, classes=self.labels_ids_list_whitelist,
                                         max_det=self.max_det)  # this takes 99% of the time

            objects = results[0].boxes
            # Generate the boxes, confidences, and classIDs
            boxes, confidences, class_ids = self.generate_boxes_confidences_classids(objects, height, width, confidence)

            # Apply Non-Maxima Suppression to suppress overlapping bounding boxes  # idxs = cv.dnn.NMSBoxes(boxes, confidences, confidence, threshold)

        if boxes is None or confidences is None or class_ids is None:
            raise "[ERROR] Required variables are set to None before drawing boxes on images."

        obstacles = []
        if len(boxes) > 0:
            for i in range(len(boxes)):
                obstDetected = {"label": class_ids[i], "confidence": confidences[i], "box": boxes[i], }
                obstacles.append(obstDetected)

        if draw_boxes:
            img_obj_det_annotated = self.draw_labels_and_boxes(img, boxes, confidences, class_ids)
        return img_obj_det_annotated, obstacles


if __name__ == '__main__':
    pass
