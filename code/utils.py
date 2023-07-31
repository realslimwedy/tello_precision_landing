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


def calculate_avg_of_tuples(values):
    # Calculate the average of the two values in the input list
    avg_num1 = int(sum(pair[0] for pair in values) / len(values))
    avg_num2 = int(sum(pair[1] for pair in values) / len(values))
    return avg_num1, avg_num2


def rolling_average_of_tuples(list_of_tuples, new_tuple, number_of_values=5):
    number_of_values = number_of_values
    if new_tuple is not None:
        list_of_tuples.append(new_tuple)  # Add the new input to the list

        # Ensure that we only consider the last N entries
        if len(list_of_tuples) > number_of_values:
            list_of_tuples = list_of_tuples[-number_of_values:]

        # Calculate the new average using the updated list
        avg_values = calculate_avg_of_tuples(list_of_tuples)

        return avg_values, list_of_tuples

    else:
        return None, list_of_tuples



def calculate_avg_of_float_values(values):
    # Calculate the average of the values in the input list
    avg_value = sum(values) / len(values)
    return avg_value


def rolling_average_of_float_values(list_of_float_values, new_float_value, number_of_values=5):
    number_of_values = number_of_values
    list_of_float_values.append(new_float_value)  # Add the new input to the list

    # Ensure that we only consider the last N entries
    if len(list_of_float_values) > number_of_values:
        list_of_float_values = list_of_float_values[-number_of_values:]

    # Calculate the new average using the updated list
    avg_value = calculate_avg_of_float_values(list_of_float_values)

    return avg_value, list_of_float_values


# define function to take an image and return the image with a green dot in the middle
def add_central_dot(image, radius=5, color=(0, 0, 0), thickness=-1):
    radius = radius
    color = color
    thickness = thickness
    image = cv.circle(image, (int(image.shape[1] / 2), int(image.shape[0] / 2)), radius=radius, color=color,
                      thickness=thickness)
    return image

def print_time_ms(name, time):
    print(f'{name}: {round(time * 1000, 1)} ms')

def print_interval_ms(name, start_time, end_time):
    print(f'{name}: {round((end_time - start_time) * 1000, 1)} ms' )

def print_fps(name, start_time, end_time):
    print(f'{name}: {round(1 / (end_time - start_time), 1)} FPS')

def seconds_within_current_clearance_period(list_of_position_clearance_timestamps, current_time):
    '''
    this function takes a list of tuples of the form (timestamp, position_within_tolerance)
    and returns the number of seconds since the first time the position was within tolerance
    and the current time
    '''
    if len(list_of_position_clearance_timestamps) > 0:
        return current_time - list_of_position_clearance_timestamps[0][0]
    else:
        return 0




if __name__ == "__main__":
    find_camera_index()
