import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest

controller = None

#Controller class for drone communication with mavros. Replaces the tello drone package for PX4 drones
class MavrosController:

    def state_cb(msg):
        global controller
        controller.current_state = msg

    def battery_cb(msg):
        global controller
        controller.battery_status = msg

    def pose_cb(msg):
        global controller
        controller.pose = msg


    def __init__(self):
        self.pose_sub = PoseStamped()
        global controller
        controller = self
        self.rate = None
        self.set_mode_client = None
        self.arming_client = None
        self.local_pos_pub = None
        self.current_state = State()
        self.battery_status =
        self.state_sub = None

    def connect(self):
        self.current_state = State()
        rospy.init_node("offb_node_py")
        self.state_sub = rospy.Subscriber("mavros/state", State, callback=self.state_cb)
        self.pose_sub = rospy.Subscriber("mavros/local_position/pose",PoseStamped, callback = self.pose_cb)
        self.local_pos_pub = rospy.Publisher("mavros/setpoint_position/local", PoseStamped, queue_size=10)
        rospy.wait_for_service("/mavros/cmd/arming")
        self.arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)
        rospy.wait_for_service("/mavros/set_mode")
        self.set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)
        # Setpoint publishing MUST be faster than 2Hz
        self.rate = rospy.Rate(20)
        # Wait for Flight Controller connection
        while (not rospy.is_shutdown() and not self.current_state.connected):
            self.rate.sleep()
        ffb_set_mode = SetModeRequest()
        ffb_set_mode.custom_mode = 'OFFBOARD'
        arm_cmd = CommandBoolRequest()
        arm_cmd.value = True
        if self.arming_client.call(arm_cmd).success == True:
            rospy.loginfo("armed")

    def get_battery(self):
        return "TODO"

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
        altitude_cm = self.me.get_distance_tof()  # in cm
        return battery_level, temperature, flight_time, barometer, altitude_cm


