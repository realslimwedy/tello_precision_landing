import math
import numpy as np
import cv2 as cv
from scipy.spatial import distance
from scipy.ndimage import gaussian_filter
from ultralytics import YOLO
import time

from .labels import labels_yolo, risk_table
from .yolo_obj_det_util import ObjectDetector
from .yolo_seg_util import SegmentationEngine


class LzFinder:
    def __init__(self, model_obj_det, model_seg, label_list=['apple'], res=(640, 480), use_seg=True, r_landing_factor=6,
                 stride=75):
        self.res = res
        self.width, self.height = res[0], res[1]
        self.posOb = [0, 0, 0, 0]
        self.use_seg = use_seg
        r_landing_factor = r_landing_factor
        self.r_landing = int(self.width / r_landing_factor)
        self.stride = stride
        self.labels = {key: value for key, value in labels_yolo.items() if key in label_list}
        self.label_ids = list(self.labels.values())
        self.obstacles = []

        self.objectDetector = ObjectDetector(model_obj_det, self.labels, self.label_ids)
        print("Object detector loaded")
        self.segEngine = SegmentationEngine(model_seg, self.label_ids)
        print("Segmentation engine loaded")

    def __str__(self):
        return f"""
Resolution: {self.res}
R-Landing: {self.r_landing}
Stride: {self.stride}
Use Segmentation: {self.use_seg}
Labels: {self.labels}"""

    def get_final_lz(self, img):
        landing_zone_xy = ()

        _, objs = self.objectDetector.infer_image(height=self.height, width=self.width, img=img, drawBoxes=True)

        if self.use_seg:
            segImg = self.segEngine.inferImage(img)
        else:
            segImg = self.segEngine.inferImageDummy(img)

        self.obstacles = []

        for obstacle in objs:
            self.posOb = obstacle.get("box")

            w, h = self.posOb[2], self.posOb[3]
            diagonal = math.sqrt(w ** 2 + h ** 2)
            minDist = int(diagonal / 2)
            self.obstacles.append([int(self.posOb[0] + w / 2), int(self.posOb[1] + h / 2), minDist])

        lzs_ranked, risk_map = self.get_ranked_lz(self.obstacles, img, segImg)

        if not lzs_ranked:
            if objs:
                print("No landing zone found, but there are objects")
            else:
                lz = {"confidence": 1,
                               "radius": self.r_landing,
                               "position": (int(self.width / 2), int(self.height / 2)),
                               "id": 0}
                lzs_ranked.append(lz)

        landing_zone = lzs_ranked[-1]
        landing_zone_xy = landing_zone["position"]

        img = self.draw_lzs_obs(lzs_ranked[-1:], self.obstacles, img)  # if no lz, nothing is drawn
        return landing_zone_xy, img, risk_map

    def get_ranked_lz(self, obstacles, img, segImg):

        lzs = self._get_landing_zones_proposals(img, self.obstacles)

        if not self.use_seg:
            risk_map = np.zeros(segImg.shape, np.uint8)
            lzs_ranked = self._rank_lzs(lzs, risk_map, self.obstacles)
        elif self.use_seg:
            risk_map = self._get_risk_map(segImg)
            lzs_ranked = self._rank_lzs(lzs, risk_map, self.obstacles)

        return lzs_ranked, risk_map

    def _dist_to_obs(self, lz, obstacles, img):
        posLz = lz.get("position")
        norm_dists = []
        if not self.obstacles:
            return 0
        else:
            for ob in self.obstacles:
                dist = self.getDistance(img, (ob[0], ob[1]), posLz)
                norm_dists.append(1 - dist)
            return np.mean(norm_dists)

    def _meets_min_safety_requirement(cls, zone_proposed, obstacles_list):
        posLz = zone_proposed.get("position")
        radLz = zone_proposed.get("radius")
        for obstacle in obstacles_list:
            touch = cls.circles_intersect(posLz[0], obstacle[0], posLz[1], obstacle[1], radLz, obstacle[2])
            if touch < 0:
                return False
        return True

    def _get_landing_zones_proposals(self, image, obstacles):
        zones_proposed = []

        for y in range(self.r_landing, image.shape[0] - self.r_landing, self.stride):
            for x in range(self.r_landing, image.shape[1] - self.r_landing, self.stride):
                lzProposed = {"confidence": math.nan, "radius": self.r_landing, "position": (x, y), "id": id, }
                if not self._meets_min_safety_requirement(lzProposed, self.obstacles):
                    lzProposed["confidence"] = 0  # NaN means safe, zero means unsafe
                zones_proposed.append(lzProposed)

        return zones_proposed

    def _get_risk_map(self, seg_array, gaussian_sigma=25):
        seg_array_float32 = seg_array.astype("float32")  # Convert seg_img to float32
        risk_array = seg_array_float32.copy()  # Make a copy of the image to use for risk_array

        for label in self.labels:
            risk_value = np.float32(risk_table[label].value)
            risk_array = np.where(risk_array == self.labels[label], risk_value, risk_array)

        risk_array = gaussian_filter(risk_array, sigma=gaussian_sigma)
        risk_array = (risk_array / 100) * 255
        risk_array = np.uint8(risk_array)

        return risk_array  # returns a risk map with values from 0 to 255

    def _risk_map_eval_basic(self, crop_array, areaLz):
        maxRisk = areaLz * 255
        cropRisk = np.sum(crop_array)
        return 1 - (cropRisk / maxRisk)

    def _rank_lzs(self, lzsProposals, riskMap, obstacles, weightDist=0, weightRisk=0, weightOb=15):

        ranked_lzs = []

        if self.use_seg == False:
            weightRisk = 0

        for lz in lzsProposals:  # this loop writes new confidence values to the lzsProposals list elements

            lzRad = lz.get("radius")
            lzPos = lz.get("position")

            if self.use_seg == True:
                mask = np.zeros_like(riskMap)  # this is a numpy array of zeros with the same shape as riskMap
                mask = cv.circle(mask, (lzPos[0], lzPos[1]), lzRad, (255, 255, 255), -1)  # -1 means filled
                # cirlce drawn on mask, green, filled

                areaLz = math.pi * lzRad * lzRad

                crop = cv.bitwise_and(riskMap, mask)  # this leaves only the circle in the risk map
                # each pixel contains the risk value (0-255) of the risk map

            riskFactor, distanceFactor, obFactor = 0, 0, 0

            if weightRisk != 0:
                if self.use_seg == True:
                    riskFactor = self._risk_map_eval_basic(crop, areaLz)  # higher value means lower risk

            if weightDist != 0:
                distanceFactor = self.getDistanceCenter(riskMap,
                                                        (lzPos[0], lzPos[1]))  # higher value means closer to center
            if weightOb != 0:
                obFactor = self._dist_to_obs(lz, self.obstacles, riskMap)  # higher value means further from self.obstacles

            if math.isnan(lz["confidence"]):  # confidence actually means how well the lz meets the requirements
                # Calculate the confidence value and set it in the lz dictionary
                total_weight = weightRisk + weightDist + weightOb
                lz["confidence"] = abs(
                    (weightRisk * riskFactor + weightDist * distanceFactor + weightOb * obFactor) / total_weight)

            if lz["confidence"] != 0:
                ranked_lzs.append(lz)

        lzsSorted = sorted(ranked_lzs, key=lambda k: k["confidence"])

        return lzsSorted

    def getDistanceCenter(self, img, pt):
        dim = img.shape
        furthestDistance = math.hypot(dim[0] / 2, dim[1] / 2)
        dist = distance.euclidean(pt, [dim[0] / 2, dim[1] / 2])
        return 1 - abs(dist / furthestDistance)  # higher value means closer to center

    def getDistance(self, img, pt1, pt2):
        dim = img.shape
        furthestDistance = math.hypot(dim[0], dim[1])
        dist = distance.euclidean(pt1, pt2)
        return 1 - abs(dist / furthestDistance)  # 1 minus because we want to maximise the distance???

    @classmethod
    def draw_lzs_obs(cls, list_lzs, list_obs, img, thickness=2):
        for obstacle in list_obs:
            cv.circle(img, (obstacle[0], obstacle[1]), obstacle[2], (0, 0, 255), thickness=thickness, )
        for lz in list_lzs:
            posLz = lz.get("position")
            radLz = lz.get("radius")
            cv.circle(img, (posLz[0], posLz[1]), radLz, (0, 255, 0), thickness=thickness)
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




if __name__ == "__main__":


    label_list = ["apple", "banana", "background", "book", "person"]
    labels = {key: value for key, value in labels_yolo.items() if key in label_list}
    label_ids = list(labels.values())

    model1 = YOLO('yoloV8_models/yolov8n.pt')
    model2 = YOLO('yoloV8_models/yolov8n-seg.pt')

    label_list = ["apple", "banana", "background", "book", "person"]

    lz_finder = LzFinder(model_obj_det=model1, model_seg=model2,res=(640, 480), label_list=label_list, use_seg=False, r_landing_factor=6,
                         stride=100)  # 640, 480 vs. 320, 240

    print(lz_finder)

    cap = cv.VideoCapture(0)


    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.resize(frame, lz_finder.res)

        landing_zone_xy, img, risk_map = lz_finder.get_final_lz(frame)

        print(landing_zone_xy)
        cv.imshow("Landing Zone", img)
        #cv.imshow("Risk Map", risk_map)

        end_time = time.time()
        iteration_time = (end_time - start_time) * 1000

        print("Iteration Time (ms):", iteration_time)

        if cv.waitKey(1) == ord('q'):
            break
