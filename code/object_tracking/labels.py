from enum import Enum

from utils import bidict

labelsYolo={'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3, 'airplane': 4, 'bus': 5, 'train': 6, 'truck': 7, 'boat': 8, 'traffic light': 9, 'fire hydrant': 10, 'stop sign': 11, 'parking meter': 12, 'bench': 13, 'bird': 14, 'cat': 15, 'dog': 16, 'horse': 17, 'sheep': 18, 'cow': 19, 'elephant': 20, 'bear': 21, 'zebra': 22, 'giraffe': 23, 'backpack': 24, 'umbrella': 25, 'handbag': 26, 'tie': 27, 'suitcase': 28, 'frisbee': 29, 'skis': 30, 'snowboard': 31, 'sports ball': 32, 'kite': 33, 'baseball bat': 34, 'baseball glove': 35, 'skateboard': 36, 'surfboard': 37, 'tennis racket': 38, 'bottle': 39, 'wine glass': 40, 'cup': 41, 'fork': 42, 'knife': 43, 'spoon': 44, 'bowl': 45, 'banana': 46, 'apple': 47, 'sandwich': 48, 'orange': 49, 'broccoli': 50, 'carrot': 51, 'hot dog': 52, 'pizza': 53, 'donut': 54, 'cake': 55, 'chair': 56, 'couch': 57, 'potted plant': 58, 'bed': 59, 'dining table': 60, 'toilet': 61, 'tv': 62, 'laptop': 63, 'mouse': 64, 'remote': 65, 'keyboard': 66, 'cell phone': 67, 'microwave': 68, 'oven': 69, 'toaster': 70, 'sink': 71, 'refrigerator': 72, 'book': 73, 'clock': 74, 'vase': 75, 'scissors': 76, 'teddy bear': 77, 'hair drier': 78, 'toothbrush': 79}


labelsAero = {
    "background": 0,
    "person": 1,
    "bike": 2,
    "car": 3,
    "drone": 4,
    "boat": 5,
    "animal": 6,
    "obstacle": 7,
    "construction": 8,
    "vegetation": 9,
    "road": 10,
    "sky": 11,
}

labelsGraz = {
    "unlabeled": 0,
    "pavedArea": 1,
    "gravel": 2,
    "grass": 3,
    "dirt": 4,
    "water": 5,
    "rocks": 6,
    "pool": 7,
    "lowVegetation": 8,
    "roof": 9,
    "wall": 10,
    "window": 11,
    "door": 12,
    "fence": 13,
    "fencePole": 14,
    "person": 15,
    "animal": 16,
    "car": 17,
    "bike": 18,
    "tree": 19,
    "baldTree": 20,
    "arMarker": 21,
    "obstacle": 22,
    "conflicting": 23,
}


class RiskLevel(Enum):
    VERY_HIGH = 100
    HIGH = 20
    MEDIUM = 10
    LOW = 5
    ZERO = 0


# Risk table for the safe landing zone finder
risk_table = {
    "unlabeled": RiskLevel.ZERO,
    "pavedArea": RiskLevel.LOW,
    "grave": RiskLevel.LOW,
    "grass": RiskLevel.ZERO,
    "dirt": RiskLevel.LOW,
    "water": RiskLevel.HIGH,
    "rocks": RiskLevel.MEDIUM,
    "pool": RiskLevel.HIGH,
    "lowVegetation": RiskLevel.ZERO,
    "roof": RiskLevel.HIGH,
    "wall": RiskLevel.HIGH,
    "window": RiskLevel.HIGH,
    "door": RiskLevel.HIGH,
    "fence": RiskLevel.HIGH,
    "fencePole": RiskLevel.HIGH,
    "person": RiskLevel.VERY_HIGH,
    "animal": RiskLevel.VERY_HIGH,
    "car": RiskLevel.VERY_HIGH,
    "bike": RiskLevel.VERY_HIGH,
    "tree": RiskLevel.HIGH,
    "baldTree": RiskLevel.HIGH,
    "arMarker": RiskLevel.ZERO,
    "obstacle": RiskLevel.HIGH,
    "conflicting": RiskLevel.HIGH,
    "background": RiskLevel.ZERO,
    "drone": RiskLevel.MEDIUM,
    "boat": RiskLevel.MEDIUM,
    "construction": RiskLevel.HIGH,
    "vegetation": RiskLevel.LOW,
    "road": RiskLevel.ZERO,
    "sky": RiskLevel.VERY_HIGH,
}

# Safety indicator for the metrics module. This is the Ground truth.
notSafe = [
    "person",
    "water",
    "fence",
    "fencePole",
    "bike",
    "animal",
    "car",
    "bicycle",
    "tree",
    "baldTree",
    "pool",
    "wall",
    "door",
    "drone",
    "construction",
    "boat"
]

biLabelsAero = bidict(labelsAero)
biLabelsGraz = bidict(labelsGraz)
biLabelsYolo = bidict(labelsYolo)

datasetLabels = {"aeroscapes": biLabelsAero, "graz": biLabelsGraz, "yolo": biLabelsYolo}
