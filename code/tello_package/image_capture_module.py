import datetime
import time
import os

import cv2 as cv


def ImageCapture():
    def __init__(self, res, fps_video_recording=30):
        self.res = res
        self.fps_video_recording = fps_video_recording

    def save_image(self, frame, img_num, path):
        self.frame = frame
        self.path = path

        if img_num == 1:
            new_path = os.path.join(path, f'image_batch_{time.time()}')
            os.mkdir(new_path)
        img_name = os.path.join(new_path, f'{str(uuid.uuid1())}.jpg')

        cv.imwrite(img_name, frame)

        img_num += 1
        last_time = time.time()

        return img_num, path, last_time

    def record_video(self, frame, out):
        if out == None:
            filename = os.path.join('..', 'data', 'videos_saved_by_drone', f'video_{datetime.now()}.mp4')
            frames_per_second = self.fps_video_recording
            video_codec = cv.VideoWriter_fourcc(*'mp4v')

            out = cv.VideoWriter(filename, video_codec, frames_per_second, self.res)

        out.write(frame)

        return out
