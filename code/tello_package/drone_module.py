from djitellopy import tello
import time
from code.config import tello_wifi
from code.utils import connect_to_wifi
import cv2 as cv


class Drone():
    def __init__(self, res=(640, 480), mirror_down=True, speed=50):
        self.resolution = res
        self.mirror_down = mirror_down
        self.drone_is_on = False
        self.flight_phase = 'TAKEOFF'
        self.me = tello.Tello()
        self.speed = speed

    def __repr__(self):
        return f'''
Drone object with resolution {self.resolution}
mirror_down: {self.mirror_down}
drone_is_on: {self.drone_is_on}
flight_phase: {self.flight_phase}'''

    def power_up(self):
        connect_to_wifi(tello_wifi)
        self.me.connect()
        self.drone_is_on = True
        print('#######################################################')
        print('Drone POWER-UP Sequence:')
        print(f'Drone is on: {self.drone_is_on}')
        print(f'Battery Level: {self.me.get_battery()} %')
        print(f'Flight Phase: {self.flight_phase}')
        print('#######################################################')
        self.me.streamon()

    def power_down(self):
        print('#######################################################')
        print('Drone POWER-DOWN Sequence:')
        print(f'Remaining Battery Level: {self.me.get_battery()} %')
        if self.me.is_flying == True:
            self.me.land()
        self.me.streamoff()
        self.me.end()
        print(f'Drone powered down and disconnected from computer')
        print('#######################################################')

    def get_flight_state(self):
        alt = self.me.get_height()
        speed_x = self.me.get_speed_x()
        speed_y = self.me.get_speed_y()
        speed_z = self.me.get_speed_z()
        acc_x = self.me.get_acceleration_x()
        acc_y = self.me.get_acceleration_y()
        acc_z = self.me.get_acceleration_z()
        roll = self.me.get_roll()
        pitch = self.me.get_pitch()
        yaw = self.me.get_yaw()
        return alt, speed_x, speed_y, speed_z, acc_x, acc_y, acc_z, roll, pitch, yaw

    def get_drone_sensor_data(self):
        flight_time = self.me.get_flight_time()
        battery_level = self.me.get_battery()  # in %
        temperature = self.me.get_temperature()  # in °C
        barometer = self.me.get_barometer()  # in Pascal or in meter?
        distance_tof = self.me.get_distance_tof()  # in cm
        return battery_level, temperature, flight_time, barometer, distance_tof

    def get_frame(self):
        img = self.me.get_frame_read().frame
        img = cv.resize(img, self.resolution)
        if self.mirror_down:
            img = cv.flip(img, 0)
        # colors are BGR but will be converted to RGB in main.py
        return img


if __name__ == "__main__":
    drone = Drone()
    drone.power_up()

    while True:
        img = drone.get_frame()
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        cv.imshow('Video Feed', img)
        print(drone.get_drone_sensor_data()[0])
        if cv.waitKey(1) & 0xFF == ord('q'):
            drone.power_down()
            break
