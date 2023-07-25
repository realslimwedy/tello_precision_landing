# capturing_module.py

import cv2
import time

def save_image(img):
    filename=f'data/images_saved_by_drone/{time.time()}.jpg'
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(filename, img)

def save_video(img,output,height,width):
    if output == None:
        filename=f'data/videos_saved_by_drone/{time.time()}.mp4'
        frames_per_second=30
        height, width = height, width
        video_codec = cv2.VideoWriter_fourcc(*'mp4v')
        output = cv2.VideoWriter(filename, video_codec, frames_per_second, (width, height)) 

    output.write(img)
    
    return output