import cv2
import time

def save_image(img):
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f'data/images_saved_by_drone/{time.time()}.jpg', img)
    time.sleep(0.2)
