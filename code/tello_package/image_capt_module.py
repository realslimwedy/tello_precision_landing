import datetime
import time
import os
import uuid

import cv2 as cv


class ImageCapture:

    def __init__(self, resolution=(640, 480), fps_video_recording=30):
        self.resolution = resolution
        self.fps_video_recording = fps_video_recording

    def save_image(self, frame, img_num, img_saving_path):
        if img_num == 1:
            img_saving_path = os.path.join('..','data','images_saved_by_drone', f'image_batch_{time.time()}')
            os.mkdir(img_saving_path)
        img_name = os.path.join(img_saving_path, f'{str(uuid.uuid1())}.jpg')

        frame = cv.flip(frame, 1)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        cv.imwrite(img_name, frame)

        img_num += 1
        last_img_time = time.time()

        return img_num, img_saving_path, last_img_time

    def record_video(self, frame, out):
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = cv.flip(frame, 1)
        if out == None:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = os.path.join('..', 'data', 'videos_saved_by_drone', f'video_{current_time}.mp4')
            frames_per_second = self.fps_video_recording
            video_codec = cv.VideoWriter_fourcc(*'mp4v')

            out = cv.VideoWriter(filename, video_codec, frames_per_second, self.resolution)

        out.write(frame)

        return out
