from ultralytics import YOLO
import cv2 as cv


class ObjectDetector():
    def __init__(self, model_path, labels_dic_filtered, verbose=False, max_det=None, conf_thres_obj_det = 0.25):
        self.model = YOLO(model_path)
        self.labels_dic_filtered = labels_dic_filtered
        self.max_det = max_det
        self.verbose = verbose
        self.labels_ids_list_whitelist = list(labels_dic_filtered.values())
        self.labels_dic_filtered_inverted = {value: key for key, value in labels_dic_filtered.items()}
        self.color = [0, 0, 0]  # black
        self.conf_thres_obj_det = conf_thres_obj_det

    def __str__(self):
        return f"ObjectDetector instance with model: {self.model}"

    def infer_image(self, img, draw_boxes=False):
        '''
        - Function returns the labelled image and the obstacles detected
        - obstacles are returned as a list of dictionaries
        - an obstacle consists of an:
            - bounding box
            - confidence
            - id
        '''

        obstacles_rectangles_list = []

        # YOLO MODEL INFERENCE #########################################################################################
        yolo_results = self.model.predict(img, verbose=self.verbose, classes=self.labels_ids_list_whitelist,
                                          max_det=self.max_det)
        yolo_objects = yolo_results[0].boxes

        # GENERATE RECTANGLE, CONFIDENCE, CLASS-IDS LISTS ##############################################################
        rect_list, conf_list, cls_ids_list = self.gen_rect_conf_clas_ids_lists(yolo_objects, self.conf_thres_obj_det)

        # FILL OBSTACLES LIST ##########################################################################################
        if len(rect_list) > 0:
            for i in range(len(rect_list)):
                obstacle = {"label": cls_ids_list[i], "confidence": conf_list[i], "box": rect_list[i], }
                obstacles_rectangles_list.append(obstacle)

        # DRAW OBSTACLES ON IMAGE ######################################################################################
        if draw_boxes:
            img = self.draw_labels_and_boxes(img, rect_list, conf_list, cls_ids_list)

        return img, obstacles_rectangles_list

    def gen_rect_conf_clas_ids_lists(self, yolo_objects, conf_thres):

        rect_list = []
        conf_list = []
        cls_ids_list = []

        for obj in yolo_objects:
            xywh = obj.xywh.tolist()[0]
            conf = obj.conf.tolist()[0]
            cls = int(obj.cls.tolist()[0])

            if conf > conf_thres:
                if cls in self.labels_dic_filtered_inverted:
                    x = int(xywh[0] - (xywh[2] / 2))
                    y = int(xywh[1] - (xywh[3] / 2))

                    rect_list.append([x, y, xywh[2], xywh[3]])
                    conf_list.append(float(conf))
                    cls_ids_list.append(cls)

        return rect_list, conf_list, cls_ids_list

    def draw_labels_and_boxes(self, img, rect_list, conf_list, cls_ids_list):

        if len(rect_list) > 0:
            for i in range(len(rect_list)):
                # Get the bounding box coordinates
                x, y = int(rect_list[i][0]), int(rect_list[i][1])
                w, h = int(rect_list[i][2]), int(rect_list[i][3])

                # Draw the bounding box rectangle and label on the image
                cv.rectangle(img, (x, y), (x + w, y + h), self.color, 2)

                # Write class as text next inside the rectangle
                text_x = x + 5
                text_y = y + 20
                text = "{}: {:4f}".format(self.labels_dic_filtered_inverted[cls_ids_list[i]], conf_list[i])
                cv.putText(img, text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 2)

        return img


if __name__ == '__main__':
    pass
