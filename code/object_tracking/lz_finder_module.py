import math
import numpy as np
import cv2 as cv
from scipy.spatial import distance
from scipy.ndimage import gaussian_filter
from ultralytics import YOLO

from .labels import labelsYolo, risk_table
from .yolo_obj_det_util import ObjectDetector
from .yolo_seg_util import SegmentationEngine


class LzFinder:
    def __init__(self, model_obj_det, model_seg, label_list=['apple'], res=(640, 480), use_seg=True, r_landing_factor=8, stride=75):
        self.res = res
        self.width, self.height = res[0], res[1]
        self.posOb = [0, 0, 0, 0]
        self.use_seg = use_seg
        self.r_landing = int(self.width / r_landing_factor)
        self.stride = stride
        self.labels = {key: value for key, value in labelsYolo.items() if key in label_list}
        self.label_ids = list(self.labels.values())

        self.objectDetector = ObjectDetector(model_obj_det, self.labels, self.label_ids)
        print("Object detector loaded")
        self.segEngine = SegmentationEngine(model_seg, self.label_ids)
        print("Segmentation engine loaded")

    def __str__(self):
        return f"""
res: {self.res}
width: {self.width}
height: {self.height}
r_landing: {self.r_landing}
stride: {self.stride}
use seg: {self.use_seg}
labels: {self.labels}
label ids: {self.label_ids}"""

    def get_final_lz(self, img):
        landing_zone_xy = ()
        _, objs = self.objectDetector.infer_image(height=self.height, width=self.width, img=img, drawBoxes=True)
        segImg = self.segEngine.inferImage(img)

        obstacles = []

        for obstacle in objs:
            self.posOb = obstacle.get("box")

            w, h = self.posOb[2], self.posOb[3]
            diagonal = math.sqrt(w ** 2 + h ** 2)
            minDist = int(diagonal/2)
            obstacles.append(
                [int(self.posOb[0] + w / 2),
                 int(self.posOb[1] + h / 2),
                 minDist]
            )

        lzs_ranked, risk_map = self.get_ranked_lz(obstacles, img, segImg)

        if lzs_ranked:
            landing_zone = lzs_ranked[-1]
            landing_zone_xy = landing_zone["position"]
            print(landing_zone_xy)  # tuple, e.g. (230, 380)
        else:
            print("No landing zone found")

        img = self.draw_lzs_obs(lzs_ranked[-1:], obstacles, img)  # if no lz, nothing is drawn
        return landing_zone_xy, img, risk_map

    def get_ranked_lz(self, obstacles, img, segImg):

        lzs = self._get_landing_zones_proposals(img, obstacles)

        if not self.use_seg:
            risk_map = np.zeros(segImg.shape, np.uint8)
            lzs_ranked = self._rank_lzs(lzs, risk_map, obstacles)
        elif self.use_seg:
            risk_map = self._get_risk_map(segImg)
            lzs_ranked = self._rank_lzs(lzs, risk_map, obstacles)

        return lzs_ranked, risk_map

    def _dist_to_obs(self, lz, obstacles, img):
        posLz = lz.get("position")
        norm_dists = []
        if not obstacles:
            return 0
        else:
            for ob in obstacles:
                dist = self.getDistance(img, (ob[0], ob[1]), posLz)
                norm_dists.append(1 - dist)
            return np.mean(norm_dists)

    def _meets_min_safety_requirement(cls, zone_proposed, obstacles_list):
        """Checks if a proposed safety zone is breaking the min. safe distance of all the high-risk obstacles detected in an image

        Args:
            zone_proposed (tuple): coordinates of the proposed zone in the x,y,r_landing format
            obstacles_list (list of tuples): list of coordinates of the high-risk obstacles in the x,y,r_min_safe_dist format

        Returns:
            (bool): True if it meets safety req., False otherwise.
        """
        posLz = zone_proposed.get("position")
        radLz = zone_proposed.get("radius")
        for obstacle in obstacles_list:
            touch = cls.circles_intersect(
                posLz[0], obstacle[0], posLz[1], obstacle[1], radLz, obstacle[2]
            )
            if touch < 0:
                return False
        return True

    def _get_landing_zones_proposals(self, image, high_risk_obstacles):
        """Returns list of lzs proposal based that meet the
        min safe distance of all the high risk obstacles

        :param high_risk_obstacles: tuple in the following format (x,y,min_safe_dist)
        :type high_risk_obstacles: tuple
        :param stride: how much stride between the proposed regions.
        :type stride: int
        :param r_landing: min safe landing radius - size of lz in pixels
        :type r_landing: int
        :param image: image to find lzs on
        :type image: Mat
        :return: list of lzs in the lz format
        :rtype: lz
        """
        zones_proposed = []

        for y in range(self.r_landing, image.shape[0] - self.r_landing, self.stride):
            for x in range(self.r_landing, image.shape[1] - self.r_landing, self.stride):
                lzProposed = {
                    "confidence": math.nan,
                    "radius": self.r_landing,
                    "position": (x, y),
                    "id": id,
                }
                if not self._meets_min_safety_requirement(
                        lzProposed, high_risk_obstacles
                ):
                    lzProposed["confidence"] = 0  # NaN means safe, zero means unsafe
                zones_proposed.append(lzProposed)

        return zones_proposed

    def _get_risk_map(self, seg_array, gaussian_sigma=25):
        '''
        Obtain a risk map based on the segmentation image with values from 0 to 255
        :param seg_img: each pixel carries the numeric class label
        '''
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
        """Evaluate normalised risk in a lz

        :param img: risk map containing pixels between 0 (low risk) and 255 (high risk)
        :type img: Mat
        :param areaLz: area of proposed lz
        :type areaLz: float
        :return: normalised risk [0.0, 1.0]
        :rtype: float
        """
        maxRisk = areaLz * 255
        cropRisk = np.sum(crop_array)
        return 1 - (cropRisk / maxRisk)

    def _rank_lzs(self, lzsProposals, riskMap, obstacles, weightDist=5, weightRisk=15, weightOb=10):

        ranked_lzs = []

        for lz in lzsProposals:  # this loop writes new confidence values to the lzsProposals list elements

            lzRad = lz.get("radius")
            lzPos = lz.get("position")
            mask = np.zeros_like(riskMap)  # this is a numpy array of zeros with the same shape as riskMap
            mask = cv.circle(mask, (lzPos[0], lzPos[1]), lzRad, (255, 255, 255), -1)  # -1 means filled
            # cirlce drawn on mask, green, filled

            areaLz = math.pi * lzRad * lzRad

            crop = cv.bitwise_and(riskMap, mask)  # this leaves only the circle in the risk map
            # each pixel contains the risk value (0-255) of the risk map

            riskFactor, distanceFactor, obFactor = 0, 0, 0

            if weightRisk != 0:
                riskFactor = self._risk_map_eval_basic(crop, areaLz)  # higher value means lower risk
            if weightDist != 0:
                distanceFactor = self.getDistanceCenter(riskMap,
                                                        (lzPos[0], lzPos[1]))  # higher value means closer to center
            if weightOb != 0:
                obFactor = self._dist_to_obs(lz, obstacles, riskMap)  # higher value means further from obstacles

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
        """Finds Normalised Distance between a given point and center of a frame

        :param img: image where the point resides
        :type img: Mat
        :param pt: coordinates of point in the form (x,y)
        :type pt: tuple
        :return: distance
        :rtype: float
        """
        dim = img.shape
        furthestDistance = math.hypot(dim[0] / 2, dim[1] / 2)
        dist = distance.euclidean(pt, [dim[0] / 2, dim[1] / 2])
        return 1 - abs(dist / furthestDistance)  # higher value means closer to center

    def getDistance(self, img, pt1, pt2):
        """Finds Normalised Distance between a two points

        :param img: image where the point resides
        :type img: Mat
        :param pt: coordinates of point in the form (x,y)
        :type pt: tuple
        :return: distance
        :rtype: float
        """
        dim = img.shape
        furthestDistance = math.hypot(dim[0], dim[1])
        dist = distance.euclidean(pt1, pt2)
        return 1 - abs(dist / furthestDistance)  # 1 minus because we want to maximise the distance???

    @classmethod
    def draw_lzs_obs(cls, list_lzs, list_obs, img, thickness=2):
        """Adds annotation on image and landing zone proposals for visualisation

        :param list_lzs: list of lzs int the lz data struct
        :type list_lzs: list
        :param list_obs: list of obstacles in the obstacle format (x,y,min_safe_dist)
        :type list_obs: list
        :param img: image to add annotation on
        :type img: Mat
        :param thickness: thickness of circles, defaults to 3
        :type thickness: int, optional
        :return: image with added annotations
        :rtype: Mat
        """
        for obstacle in list_obs:
            cv.circle(
                img,
                (obstacle[0], obstacle[1]),
                obstacle[2],
                (0, 0, 255),
                thickness=thickness,
            )
        for lz in list_lzs:
            posLz = lz.get("position")
            radLz = lz.get("radius")
            cv.circle(
                img, (posLz[0], posLz[1]), radLz, (0, 255, 0), thickness=thickness
            )
        return img

    @classmethod
    def circles_intersect(cls, x1, x2, y1, y2, r1, r2):
        """Checks if two circle intersect

        :param x1: x-coordinate of first circle center
        :type x1: int
        :param x2: x-coordinate of second circle center
        :type x2: int
        :param y1: y-coordinate of first circle center
        :type y1: int
        :param y2: y-coordinate of second circle center
        :type y2: int
        :param r1: radius of first circle
        :type r1: int
        :param r2: radius of second circle
        :type r2: int
        :return: -3 (C2 is in C1), -2 (C1 is in C2), -1 (circles intersect), 0 (circles don't intersect)
        :rtype: int
        """

        d = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        if d < r1 - r2:
            # 'print("C2  is in C1")
            return -3
        elif d < r2 - r1:
            return -2
            # print("C1  is in C2")
        elif d > r1 + r2:
            return 0
            # print("Circumference of C1  and C2  intersect")
        else:
            return -1
            # print("C1 and C2  do not overlap")


##############################################################################################################

if __name__ == "__main__":

    label_list = ["apple", "banana", "background", "book", "person"]
    lzFinder = LzFinder(res=(640, 480), label_list=label_list, use_seg=True, r_landing_factor=8, stride=75)  #640, 480 vs. 320, 240
    print(lzFinder)

    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.resize(frame, lzFinder.res)

        landing_zone_xy, img, risk_map = lzFinder.get_final_lz(frame)

        print(landing_zone_xy)
        cv.imshow("Landing Zone", img)
        #cv.imshow("Risk Map", risk_map)

        if cv.waitKey(1) == ord('q'):
            break
