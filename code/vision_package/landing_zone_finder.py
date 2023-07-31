import math
import numpy as np
import cv2 as cv
from scipy.spatial import distance
from . import yolo_obj_det_util
from . import yolo_seg_util
from . import labels


class LzFinder:
    def __init__(self, model_obj_det_path, model_seg_path, labels_dic_filtered, weight_dist, weight_risk, weight_obs,
                 max_det=None, res=(640, 480),
                 use_seg_for_lz=False, r_landing_factor=6, stride=75, verbose=False, draw_lzs=True):
        self.weight_dist = weight_dist
        self.weight_risk = weight_risk
        self.weight_obs = weight_obs
        self.draw_lzs = draw_lzs
        self.use_seg = use_seg_for_lz
        self.stride = stride
        self.labels_dic_filtered = labels_dic_filtered
        # for weights for debugging

        self.width, self.height = res
        self.r_landing = int(self.width * r_landing_factor / 100)

        self.posOb = [0, 0, 0, 0]

        self.object_detector = yolo_obj_det_util.ObjectDetector(model_path=model_obj_det_path,
                                                                labels_dic_filtered=self.labels_dic_filtered,
                                                                max_det=max_det, verbose=verbose)

        self.seg_engine = yolo_seg_util.SegmentationEngine(model_path=model_seg_path,
                                                           labels_dic_filtered=self.labels_dic_filtered,
                                                           verbose=verbose)

    def __str__(self):
        return f"""
        Resolution: {self.res}
        R-Landing: {self.r_landing}
        Stride: {self.stride}
        Use Segmentation: {self.use_seg}
        Labels: {self.labels_dic_filtered}"""





    def get_final_lz(self, img):
        objs = []
        obstacles = []
        lzs_ranked = []
        landing_zone_xy = (int(self.width / 2), int(self.height / 2))

        # get objects from YOLO object detection inference
        # unless deep copy of image, image will be annotated with bounding boxes
        img, objs = self.object_detector.infer_image(height=self.height, width=self.width, img=img,
                                                   draw_boxes=True)  # this takes around 150ms / 99% of iteration time

        # get segmentation image from YOLO segmentation inference
        if self.use_seg:
            seg_img = self.seg_engine.infer_image(img=img, width=self.width, height=self.height)
        else:
            seg_img = self.seg_engine.infer_image_dummy(img)

        # get obstacles from YOLO object detection inference
        for obstacle in objs:
            self.posOb = obstacle.get("box")
            w, h = self.posOb[2], self.posOb[3]
            diagonal = math.sqrt(w ** 2 + h ** 2)
            min_dist = int(diagonal / 2)
            obstacles.append([int(self.posOb[0] + w / 2), int(self.posOb[1] + h / 2), min_dist])

        # get lz proposals which don't intersect with obstacles ranked according to weighted score
        lzs_ranked, risk_map = self.get_ranked_lz(obstacles, seg_img)

        if objs:
            if lzs_ranked:
                # determine top lz proposal
                landing_zone = lzs_ranked[-1]

                # extract xy coordinates
                landing_zone_xy = landing_zone["position"]

                # draw top lz proposal
                img = self.draw_lzs_obs([lzs_ranked[-1]], obstacles, img, thickness=2,
                                        draw_lzs=self.draw_lzs)  # if no lz, nothing is drawn
            else:
                print("No landing zone found, objects detected")
        else:
            landing_zone_xy = (int(self.width / 2), int(self.height / 2))
            print("No landing zone found, no objects detected")

        '''lz = {"confidence": 1, "radius": self.r_landing,
              "position": (int(self.width / 2), int(self.height / 2)), "id": 1
              }
        lzs_ranked.append(lz)'''

        return landing_zone_xy, img, risk_map






    def get_ranked_lz(self, obstacles, segImg):

        lz_proposals = self._get_lz_proposals(obstacles)

        if not self.use_seg:
            risk_map = np.zeros(segImg.shape, np.uint8)
            lzs_ranked = self._rank_lzs(lz_proposals, risk_map, obstacles)
        elif self.use_seg:
            risk_map = self._get_risk_map(segImg)
            lzs_ranked = self._rank_lzs(lz_proposals, risk_map, obstacles)

        return lzs_ranked, risk_map

    def _mean_dist_to_all_obstacles(self, lz, obstacles):
        pos_lz = lz.get("position")
        dist_normalized_list = []
        if not obstacles:
            return 0
        else:
            for ob in obstacles:
                dist_normalized = self.get_distance_normalized((ob[0], ob[1]), pos_lz)
                dist_normalized_list.append(dist_normalized)
            return np.mean(dist_normalized_list)

    def _meets_min_safety_requirement(cls, zone_proposed, obstacles_list):
        pos_lz = zone_proposed.get("position")
        rad_lz = zone_proposed.get("radius")
        for obstacle in obstacles_list:
            touch = cls.circles_intersect(pos_lz[0], obstacle[0], pos_lz[1], obstacle[1], rad_lz, obstacle[2])
            if touch < 0:
                return False
        return True

    def _get_lz_proposals(self, obstacles):
        zones_proposed = []

        for y in range(self.r_landing, self.height - self.r_landing, self.stride):

            for x in range(self.r_landing, self.width - self.r_landing, self.stride):

                lz_proposed = {
                    "confidence": math.nan,
                    "radius": self.r_landing,
                    "position": (x, y),
                    "id": 1
                }

                if not self._meets_min_safety_requirement(lz_proposed, obstacles):
                    lz_proposed["confidence"] = 0  # NaN means safe, zero means unsafe
                zones_proposed.append(lz_proposed)

        return zones_proposed





    def _rank_lzs(self, lzs_proposals, risk_map, obstacles):

        ranked_lzs = []

        if not self.use_seg:
            self.weight_risk = 0


        for lz in lzs_proposals:  # this loop writes new confidence values to the lzsProposals list elements

            lz_rad = lz.get("radius")
            lz_pos = lz.get("position")

            if self.use_seg == True:
                mask = np.zeros_like(risk_map)  # this is a numpy array of zeros with the same shape as riskMap
                mask = cv.circle(mask, (lz_pos[0], lz_pos[1]), lz_rad, (255, 255, 255), -1)  # -1 means filled

                area_lz = math.pi * lz_rad * lz_rad

                crop = cv.bitwise_and(risk_map, mask)
                # this leaves only the circle in the risk map  # each pixel contains the risk value (0-255) of the risk map

            risk_factor, distance_from_center_factor, distance_from_obstacles_norm_mean_factor = 0, 0, 0

            if self.weight_risk != 0:
                if self.use_seg == True:
                    risk_factor = self._risk_map_eval_basic(crop, area_lz)  # higher value means lower risk

            if self.weight_dist != 0:
                distance_from_center_factor = self.get_distance_center(
                    (lz_pos[0], lz_pos[1]))  # higher value means closer to center

            if self.weight_obs != 0:
                distance_from_obstacles_norm_mean_factor = self._mean_dist_to_all_obstacles(lz, obstacles)
                # higher value means further from obstacles

            if math.isnan(lz["confidence"]):  # confidence actually means how well the lz meets the requirements
                # Calculate the confidence value and set it in the lz dictionary
                total_weight = self.weight_risk + self.weight_dist + self.weight_obs

                lz["confidence"] = distance_from_center_factor
                #abs((self.weight_risk * risk_factor + self.weight_dist * distance_from_center_factor + self.weight_obs * distance_from_obstacles_norm_mean_factor) / total_weight)

            if lz["confidence"] != 0:
                ranked_lzs.append(lz)


        # Check if lzs_sorted is not empty before accessing the last element
        if ranked_lzs:
            lzs_sorted = sorted(ranked_lzs, key=lambda k: k["confidence"])
        else:
            lzs_sorted = []

        return lzs_sorted



    def get_distance_center(self, pt):
        furthest_distance = math.hypot(self.height/2, self.width/2)
        dist = distance.euclidean(pt, (self.width/2, self.height/2))
        return 1 - abs(dist / furthest_distance)  # higher value means closer to center

    def get_distance_normalized(self, pt1, pt2):
        max_distance = math.hypot(self.height, self.width)
        dist = distance.euclidean(pt1, pt2)
        dist_normalized = dist / max_distance
        return dist_normalized

    @classmethod
    def draw_lzs_obs(cls, list_lzs, list_obs, img, thickness=2, draw_lzs=True):
        for obstacle in list_obs:
            cv.circle(img, (obstacle[0], obstacle[1]), obstacle[2], (255, 0, 0), thickness=thickness, )
        if draw_lzs:
            for lz in list_lzs:
                pos_lz = lz.get("position")
                rad_lz = lz.get("radius")
                cv.circle(img, (pos_lz[0], pos_lz[1]), rad_lz, (0, 255, 255), thickness=thickness)
        return img

    @classmethod
    def circles_intersect(cls, x1, x2, y1, y2, r1, r2):

        d = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        if d < r1 - r2:
            # 'print("C2  is in C1")
            return -3
        elif d < r2 - r1:
            return -2  # print("C1  is in C2")
        elif d > r1 + r2:
            return 0  # print("Circumference of C1  and C2  intersect")
        else:
            return -1  # print("C1 and C2  do not overlap")

    def calculate_average(sef, values):
        # Calculate the average of the two values in the input list
        avg_num1 = sum(pair[0] for pair in values) / len(values)
        avg_num2 = sum(pair[1] for pair in values) / len(values)
        return avg_num1, avg_num2

    def rolling_average(self, landing_zone_xy_rolling_list, new_tuple, number_of_values=5):
        number_of_values = number_of_values
        if new_tuple is not None:
            landing_zone_xy_rolling_list.append(new_tuple)  # Add the new input to the list

            # Ensure that we only consider the last N entries
            if len(landing_zone_xy_rolling_list) > number_of_values:
                landing_zone_xy_rolling_list = landing_zone_xy_rolling_list[-number_of_values:]

            # Calculate the new average using the updated list
            landing_zone_xy_avg = self.calculate_average(landing_zone_xy_rolling_list)
        else:
            landing_zone_xy_avg = None
        return landing_zone_xy_avg, landing_zone_xy_rolling_list

    def _get_risk_map(self, seg_array_with_class_id, gaussian_sigma=25):
        '''seg_array_float32 = seg_array.astype("float32")  # Convert seg_img to float32
        risk_array = seg_array_float32.copy()  # Make a copy of the image to use for risk_array'''

        risk_array_with_risk_level = seg_array_with_class_id.astype(
            "float32")  # REALLY NEEDED??? Convert seg_img to float32

        for key in self.labels_dic_filtered:
            risk_value = np.float32(labels.risk_table[key].value)
            risk_array_with_risk_level = np.where(risk_array_with_risk_level == self.labels_dic_filtered[key],
                                                  risk_value, risk_array_with_risk_level)

        # risk_array = gaussian_filter(risk_array, sigma=gaussian_sigma)
        # risk_array = (risk_array / 100) * 255
        risk_array_with_risk_level = np.uint8(risk_array_with_risk_level)

        return risk_array_with_risk_level  # returns a risk map with values from 0 to 255

    def _risk_map_eval_basic(self, crop_array, areaLz):
        maxRisk = areaLz * 255
        cropRisk = np.sum(crop_array)
        return 1 - (cropRisk / maxRisk)


if __name__ == "__main__":
    pass
