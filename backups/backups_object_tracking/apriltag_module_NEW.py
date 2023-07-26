#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import numpy as np
import cv2
from pupil_apriltags import Detector

def apriltag_center_area(image):
    # Set up the necessary variables and detector

    at_detector = Detector(
        families = 'tag36h11',
        nthreads = 1,
        quad_decimate = 2.0,
        quad_sigma = 0.0,
        refine_edges = 1,
        decode_sharpening = 0.25,
        debug = 0
    )

    # Make a copy of the image for debugging purposes
    debug_image = copy.deepcopy(image)
    
    # Convert the image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect AprilTags in the image
    tags = at_detector.detect(
        image,
        estimate_tag_pose=False,
        camera_params=None,
        tag_size=None,
    )
    
    center_x = 0
    center_y = 0
    area = 0

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
        cv2.circle(debug_image, center, 5, (0, 0, 255), 2)

        # Draw the tag outline
        cv2.line(debug_image, (corner_01[0], corner_01[1]),
                (corner_02[0], corner_02[1]), (255, 0, 0), 2)
        cv2.line(debug_image, (corner_02[0], corner_02[1]),
                (corner_03[0], corner_03[1]), (255, 0, 0), 2)
        cv2.line(debug_image, (corner_03[0], corner_03[1]),
                (corner_04[0], corner_04[1]), (0, 255, 0), 2)
        cv2.line(debug_image, (corner_04[0], corner_04[1]),
                (corner_01[0], corner_01[1]), (0, 255, 0), 2)
        
        cv2.putText(debug_image, str(tag_id), (center[0] - 10, center[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Calculate the area
        area = cv2.contourArea(np.array([corner_01, corner_02, corner_03, corner_04]))
        
        # Update the center point coordinates
        center_x = center[0]
        center_y = center[1]

    # Return the modified image and the required values
    return debug_image, [center_x, center_y], area


####################################################################################################


def main():
    cap = cv2.VideoCapture(0)  # Use the default webcam (change the index if needed)

    while True:
        ret, image = cap.read()
        if not ret:
            break

        # Call the apriltag_center_area function
        image, [center_x, center_y], area = apriltag_center_area(image)

        # Display the frame
        cv2.imshow('Webcam', image)

        # Print the tag ID, center coordinates, and area
        print("Center X:", center_x)
        print("Center Y:", center_y)
        print("Area:", area)

        key = cv2.waitKey(1)
        if key == 27:  # Press Esc to exit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()