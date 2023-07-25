import subprocess
import cv2 as cv


def connect_to_wifi(network_name):
    network_interface = "en0"  # Replace with the appropriate network interface for your Mac

    try:
        subprocess.run(["networksetup", "-setairportnetwork", network_interface, network_name], check=True)
        print(f"Connected to Wi-Fi network: {network_name}")
    except subprocess.CalledProcessError:
        print(f"Failed to connect to Wi-Fi network: {network_name}")


def find_camera_index():
    for i in range(10):  # Try a range of indices (0 to 9) to find the correct camera
        cap = cv.VideoCapture(i)
        if cap.isOpened():
            print(f"Found camera at index {i}")
            cap.release()

if __name__=="__main__":
    find_camera_index()