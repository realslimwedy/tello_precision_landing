from ultralytics import YOLO
import numpy as np


class SegmentationEngine():
    def __init__(self, model_path, labels_dic_filtered, verbose=False):
        self.model = YOLO(model_path)
        self.labels_ids_list_filtered = list(labels_dic_filtered.values())
        self.verbose = verbose
        self.background_index = 80  # workaround for risk map later as class 0 is used for "people"

    def __str__(self):
        return f"SegmentationEngine instance with model: {self.model}"

    def infer_image(self, img, height, width):

        # PREPARE VARIABLES ############################################################################################
        width, height = width, height
        seg_output_array_with_class_predictions = np.full((height, width), self.background_index, dtype=int)

        # YOLO MODEL INFERENCE #########################################################################################
        yolo_results = self.model.predict(source=img, verbose=self.verbose, classes=self.labels_ids_list_filtered)
        yolo_objects = yolo_results[0].boxes  # type: 'ultralytics.engine.results.Boxes'
        yolo_masks = yolo_results[0].masks  # type: 'ultralytics.engine.results.Masks'

        # POPULATE OUTPUT ARRAY PIXELS WITH CLASS IDS ##################################################################
        for i in range(len(yolo_objects)):
            class_id = int(yolo_objects[i].cls.tolist()[0])  # type int
            mask_list = yolo_masks.data[i].tolist()  # type list with 0s and 1s indicating where class is present

            # Get height and width from the shape of mask's tensor, background: by default yolo scales down input image
            mask_height, mask_width = yolo_masks.shape[1], yolo_masks.shape[2]

            # Calculate scaling factors
            scale_x = width / mask_width
            scale_y = height / mask_height

            for y, row in enumerate(mask_list):
                for x, val in enumerate(row):
                    if val == 1:
                        # Scale the coordinates back to the original image resolution
                        scaled_x = int(x * scale_x)
                        scaled_y = int(y * scale_y)
                        if scaled_x < width and scaled_y < height:
                            seg_output_array_with_class_predictions[scaled_y, scaled_x] = class_id

        # CONVERT ARRAY TYPE ###########################################################################################
        seg_output_array_with_class_predictions = np.array(seg_output_array_with_class_predictions, dtype=np.uint8)

        return seg_output_array_with_class_predictions

    def infer_image_dummy(self, img):
        width, height = img.shape[1], img.shape[0]
        output_array = np.full((height, width), self.background_index, dtype=np.uint8)
        return output_array


if __name__ == "__main__":
    pass
