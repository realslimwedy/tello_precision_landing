from ultralytics import YOLO
import numpy as np


class SegmentationEngine():
    def __init__(self, model_path, labels_dic_filtered, verbose=False):
        self.model = YOLO(model_path)
        self.labels_ids_list_filtered = list(labels_dic_filtered.values())
        self.verbose = verbose

        self.background_index = 80  # must be one greater than the last index in labels_yolo  # this is a workaround:  # main reason: a person is class 0 in YOLOv8, so background needs to be indexed differently  # using -1 would yield 255 during conversion into uint8 in inferImage()  # workaround: use 80, which is the index of the last class in labels_yolo +1  # self.background_index must a unique value and stated in labels.risk_labels  # for later conversion into risk_map

    def __str__(self):
        return f"SegmentationEngine instance with model: {self.model}"

    def infer_image(self, height, width, img):
        width, height = width, height

        results = self.model.predict(source=img, verbose=self.verbose,
                                     classes=self.labels_ids_list_filtered)  # this list
        boxes = results[0].boxes  # <class 'ultralytics.engine.results.Boxes'>
        masks = results[0].masks  # <class 'ultralytics.engine.results.Masks'>

        output_array_with_class_predictions = np.full((height, width), self.background_index, dtype=int)

        for i in range(len(boxes)):
            cls_i = int(boxes[i].cls.tolist()[0])  # type int
            mask_list = masks.data[i].tolist()  # type list with 0s and 1s indicating where class is present

            # Get the height and width from the shape of the masks tensor
            mask_height, mask_width = masks.shape[1], masks.shape[2]  # type int, corresponds to height, width of img

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
                            output_array_with_class_predictions[scaled_y, scaled_x] = cls_i

        output_array_with_class_predictions = np.array(output_array_with_class_predictions, dtype=np.uint8)

        return output_array_with_class_predictions

    def infer_image_dummy(self, img):
        width, height = img.shape[1], img.shape[0]
        output_array = np.full((height, width), self.background_index, dtype=np.uint8)
        return output_array


if __name__ == "__main__":
    pass