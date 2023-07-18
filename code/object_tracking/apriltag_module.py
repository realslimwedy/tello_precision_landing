#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import argparse
import numpy as np
import cv2 as cv
from pupil_apriltags import Detector

# the following is needed for the default values of the arguments
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument("--families", type=str, default='tag36h11')
    parser.add_argument("--nthreads", type=int, default=1)
    parser.add_argument("--quad_decimate", type=float, default=2.0)
    parser.add_argument("--quad_sigma", type=float, default=0.0)
    parser.add_argument("--refine_edges", type=int, default=1)
    parser.add_argument("--decode_sharpening", type=float, default=0.25)
    parser.add_argument("--debug", type=int, default=0)

    args = parser.parse_args()

    return args

####################################################################################################
# matlab pin, check how long substest takes


def apriltag_center_area(image):
    args = get_args()

    # Set up the necessary variables and detector
    families = args.families
    nthreads = args.nthreads
    quad_decimate = args.quad_decimate
    quad_sigma = args.quad_sigma
    refine_edges = args.refine_edges
    decode_sharpening = args.decode_sharpening
    debug = args.debug

    at_detector = Detector(
        families=families,
        nthreads=nthreads,
        quad_decimate=quad_decimate,
        quad_sigma=quad_sigma,
        refine_edges=refine_edges,
        decode_sharpening=decode_sharpening,
        debug=debug,
    )

    # Make a copy of the image for debugging purposes
    debug_image = copy.deepcopy(image)
    
    # for debugging
    # debug_image = cv.cvtColor(debug_image, cv.COLOR_BGR2GRAY)
    
    # Convert the image to grayscale
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Detect AprilTags in the image
    tags = at_detector.detect(
        image,
        estimate_tag_pose=False,
        camera_params=None,
        tag_size=None,
    )
    # try sleeping here
    debug_image, [center_x, center_y], area = draw_tags(debug_image, tags)

    # Return the modified image and the required values
    return debug_image, [center_x, center_y], area

####################################################################################################

def draw_tags(image, tags):
    
    center_x = None
    center_y = None
    area = None

    for tag in tags:
        tag_id = tag.tag_id      
        center = tag.center
        corners = tag.corners

        center = (int(tag.center[0]), int(tag.center[1]))
        corner_01 = (int(corners[0][0]), int(corners[0][1]))
        corner_02 = (int(corners[1][0]), int(corners[1][1]))
        corner_03 = (int(corners[2][0]), int(corners[2][1]))
        corner_04 = (int(corners[3][0]), int(corners[3][1]))

        # Draw the center
        cv.circle(image, center, 5, (0, 0, 255), 2)

        # Draw the tag outline
        cv.line(image, (corner_01[0], corner_01[1]),
                (corner_02[0], corner_02[1]), (255, 0, 0), 2)
        cv.line(image, (corner_02[0], corner_02[1]),
                (corner_03[0], corner_03[1]), (255, 0, 0), 2)
        cv.line(image, (corner_03[0], corner_03[1]),
                (corner_04[0], corner_04[1]), (0, 255, 0), 2)
        cv.line(image, (corner_04[0], corner_04[1]),
                (corner_01[0], corner_01[1]), (0, 255, 0), 2)
        
        cv.putText(image, str(tag_id), (center[0] - 10, center[1] - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv.LINE_AA)
        
        # Calculate the area
        area = cv.contourArea(np.array([corner_01, corner_02, corner_03, corner_04]))
        
        # Update the center point coordinates
        center_x = center[0]
        center_y = center[1]

        
    return image, [center_x, center_y], area




####################################################################################################


def main():
    cap = cv.VideoCapture(0)  # Use the default webcam (change the index if needed)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Call the apriltag_center_area function
        image, [center_x, center_y], area = apriltag_center_area(frame)

        # Display the frame
        cv.imshow('Webcam', image)

        # Print the tag ID, center coordinates, and area
        print("Center X:", center_x)
        print("Center Y:", center_y)
        print("Area:", area)

        key = cv.waitKey(1)
        if key == 27:  # Press Esc to exit
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()