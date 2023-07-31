import copy
import argparse
import numpy as np

import cv2 as cv
from pupil_apriltags import Detector


class ApriltagFinder:
    def __init__(self, resolution=(640, 480)):
        self.resolution = resolution

        self.families = None
        self.nthreads = None
        self.quad_decimate = None
        self.quad_sigma = None
        self.refine_edges = None
        self.decode_sharpening = None
        self.debug = None

    def get_args(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--device", type=int, default=0)
        parser.add_argument("--width", help='cap width', type=int, default=self.resolution[0])
        parser.add_argument("--height", help='cap height', type=int, default=self.resolution[1])
        parser.add_argument("--families", type=str, default='tag36h11')
        parser.add_argument("--nthreads", type=int, default=1)
        parser.add_argument("--quad_decimate", type=float, default=2.0)
        parser.add_argument("--quad_sigma", type=float, default=0.0)
        parser.add_argument("--refine_edges", type=int, default=1)
        parser.add_argument("--decode_sharpening", type=float, default=0.25)
        parser.add_argument("--debug", type=int, default=0)

        args = parser.parse_args()

        return args

    def apriltag_center_area(self, image):
        args = self.get_args()

        self.families = args.families
        self.nthreads = args.nthreads
        self.quad_decimate = args.quad_decimate
        self.quad_sigma = args.quad_sigma
        self.refine_edges = args.refine_edges
        self.decode_sharpening = args.decode_sharpening
        self.debug = args.debug

        at_detector = Detector(families=self.families, nthreads=self.nthreads, quad_decimate=self.quad_decimate,
                               quad_sigma=self.quad_sigma, refine_edges=self.refine_edges,
                               decode_sharpening=self.decode_sharpening, debug=self.debug, )

        debug_image = copy.deepcopy(image)

        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        tags = at_detector.detect(image, estimate_tag_pose=False, camera_params=None, tag_size=None, )

        debug_image, center, area = self.draw_tags(debug_image, tags)

        return debug_image, center, area

    def draw_tags(self, image, tags, ):
        center = (0,0)
        area = 0
        for tag in tags:
            tag_family = tag.tag_family
            tag_id = tag.tag_id
            center = tag.center
            corners = tag.corners

            center = (int(center[0]), int(center[1]))
            corner_01 = (int(corners[0][0]), int(corners[0][1]))
            corner_02 = (int(corners[1][0]), int(corners[1][1]))
            corner_03 = (int(corners[2][0]), int(corners[2][1]))
            corner_04 = (int(corners[3][0]), int(corners[3][1]))

            cv.circle(image, (center[0], center[1]), 5, (0, 0, 255), 2)

            cv.line(image, (corner_01[0], corner_01[1]), (corner_02[0], corner_02[1]), (255, 0, 0), 2)
            cv.line(image, (corner_02[0], corner_02[1]), (corner_03[0], corner_03[1]), (255, 0, 0), 2)
            cv.line(image, (corner_03[0], corner_03[1]), (corner_04[0], corner_04[1]), (0, 255, 0), 2)
            cv.line(image, (corner_04[0], corner_04[1]), (corner_01[0], corner_01[1]), (0, 255, 0), 2)

            area = cv.contourArea(np.array([corner_01, corner_02, corner_03, corner_04]))

            cv.putText(image, str(tag_id), (center[0] - 10, center[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255),
                       2, cv.LINE_AA)

        return image, center, area


def main():
    # Initialize the ApriltagFinder
    RES = (1280, 720)
    apriltag_finder = ApriltagFinder(resolution=RES)  # (1280, 720), (640, 480), (320, 240)

    # Open the default camera (you can change the device number if needed)
    cap = cv.VideoCapture(1)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()
        frame = cv.resize(frame, RES)
        if not ret:
            print("Error: Unable to capture frame.")
            break

        # Detect AprilTags and draw them on the frame
        debug_image, center, area = apriltag_finder.apriltag_center_area(frame)

        # Display the resulting frame with AprilTags
        cv.imshow('AprilTag Detection', debug_image)

        # Exit the loop when 'q' key is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the OpenCV windows
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
