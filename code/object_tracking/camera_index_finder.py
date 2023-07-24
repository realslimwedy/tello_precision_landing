import cv2 as cv

def find_camera_index():
    for i in range(10):  # Try a range of indices (0 to 9) to find the correct camera
        cap = cv.VideoCapture(i)
        if cap.isOpened():
            print(f"Found camera at index {i}")
            cap.release()

find_camera_index()