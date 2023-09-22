
class Mavros_drone:

    def __init__(self, res=(640, 480), speed=50):
        self.resolution = res
        self.drone_is_on = False
        self.flight_phase = 'Pre-Flight'
        self.me = Mavros_drone()
        self.speed = speed



    def power_up(self):
        self.me.connect()
        self.drone_is_on = True
        print('#######################################################')
        print('Drone POWER-UP Sequence:')
        print(f'Drone is on: {self.drone_is_on}')
        print(f'Battery Level: {self.me.get_battery()} %')
        print(f'Flight Phase: {self.flight_phase}')
        print('#######################################################')
        self.me.streamon()