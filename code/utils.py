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

def calculate_average(values):
    # Calculate the average of the two values in the input list
    avg_num1 = sum(pair[0] for pair in values) / len(values)
    avg_num2 = sum(pair[1] for pair in values) / len(values)
    return avg_num1, avg_num2

def rolling_average(list_of_tuples, new_tuple, number_of_values=5):
    number_of_values = number_of_values
    list_of_tuples.append(new_tuple)  # Add the new input to the list

    # Ensure that we only consider the last N entries
    if len(list_of_tuples) > number_of_values:
        list_of_tuples = list_of_tuples[-number_of_values:]

    # Calculate the new average using the updated list
    avg_values = calculate_average(list_of_tuples)

    return avg_values, list_of_tuples


if __name__=="__main__":
    find_camera_index()