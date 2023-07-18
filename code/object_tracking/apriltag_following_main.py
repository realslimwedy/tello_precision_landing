#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse
import numpy as np

import cv2 as cv
from pupil_apriltags import Detector


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


def main():
    # 引数解析 #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    families = args.families
    nthreads = args.nthreads
    quad_decimate = args.quad_decimate
    quad_sigma = args.quad_sigma
    refine_edges = args.refine_edges
    decode_sharpening = args.decode_sharpening
    debug = args.debug

    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Detector準備 #############################################################
    at_detector = Detector(
        families=families,
        nthreads=nthreads,
        quad_decimate=quad_decimate,
        quad_sigma=quad_sigma,
        refine_edges=refine_edges,
        decode_sharpening=decode_sharpening,
        debug=debug,
    )

    elapsed_time = 0

    while True:
        start_time = time.time()

        # カメラキャプチャ #####################################################
        ret, image = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(image)

        # 検出実施 #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        tags = at_detector.detect(
            image,
            estimate_tag_pose=False,
            camera_params=None,
            tag_size=None,
        )

        # 描画 ################################################################
        debug_image, tag_id, center_x, center_y, area = draw_tags(debug_image, tags, elapsed_time)

        if debug_image is not None and not np.all(debug_image == 0):
            # Display the image
            cv.imshow('AprilTag Detect Demo', debug_image)

        elapsed_time = time.time() - start_time

        # キー処理(ESC：終了) #################################################
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        # 画面反映 #############################################################
        cv.imshow('AprilTag Detect Demo', debug_image)

        print(tag_id, center_x, center_y, area)

    cap.release()
    cv.destroyAllWindows()


def draw_tags(
    image,
    tags,
    elapsed_time,
):
    tag_id = None
    center_x = None
    center_y = None
    area = None

    for tag in tags:
        tag_family = tag.tag_family
        tag_id = tag.tag_id

        # Modify the following code block to handle only tag_id = 0
        '''if tag_id != 0:
            continue'''

        center = (int(tag.center[0]), int(tag.center[1]))
        corners = tag.corners

        corner_01 = (int(corners[0][0]), int(corners[0][1]))
        corner_02 = (int(corners[1][0]), int(corners[1][1]))
        corner_03 = (int(corners[2][0]), int(corners[2][1]))
        corner_04 = (int(corners[3][0]), int(corners[3][1]))

        # Calculate the area
        area = cv.contourArea(np.array([corner_01, corner_02, corner_03, corner_04]))

        # Draw the center
        cv.circle(image, center, 5, (0, 0, 255), 2)

        # Draw the tag outline
        cv.line(image, corner_01, corner_02, (255, 0, 0), 2)
        cv.line(image, corner_02, corner_03, (255, 0, 0), 2)
        cv.line(image, corner_03, corner_04, (255, 0, 0), 2)
        cv.line(image, corner_04, corner_01, (255, 0, 0), 2)

        # Update the center point coordinates
        center_x = center[0]
        center_y = center[1]

    # Display the elapsed time
    cv.putText(image,
               "Elapsed Time:" + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
               cv.LINE_AA)

    return image, tag_id, center_x, center_y, area



if __name__ == '__main__':
    main()
