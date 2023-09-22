import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest



#Controller class for drone communication with mavros. Replaces the tello drone package for PX4 drones
class MavrosController:

    def __init__(self):
        self.rate = None
        self.set_mode_client = None
        self.arming_client = None
        self.local_pos_pub = None
        self.current_state = None
        self.state_sub = None

    def connect(self):
        self.current_state = State()
        rospy.init_node("offb_node_py")
        self.state_sub = rospy.Subscriber("mavros/state", State, callback=state_cb)
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

