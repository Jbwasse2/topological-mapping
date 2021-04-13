# This code will generalize around the tcp/ip comms that Mateus made.
# In order to get this to work you can either compile his branches,
# or you can ask Arun/Justin for the compiled binary.
# This code will subscribe to the topic "terra_command_x"
# and will do the appropiate actions based on the message.
# At the moment this is just vx,vy,vz,wx,wy,wz commands to the robot.
import socket
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


# Twist, Vector3 linear, Vector3 angular
# Vector3 x,y,z float64
# Creates socket with terrasentia to communicate over.
# linear in m/s, angular in rad/s
# Subscribes to twist message, sends twist message to terrasentia
class TerraComm(Node):
    def __init__(self, host="192.168.1.135", port=51717):
        super().__init__("terra_comm")
        # The controller should be at this IP
        self.host = host
        # The port that Mateus has defined the TCP/IP to comm over.
        self.port = port
        self.socket = self.create_socket()
        self.subscription = self.create_subscription(
            Twist, "terra_command_twist", self.twist_callback, 1
        )

    def twist_callback(self, msg):
        assert isinstance(msg, Twist)
        linear = msg.linear
        angular = msg.angular
        self.get_logger().info(
            "[TerraComm] Linear: {0},{1},{2} Angular: {3},{4},{5}".format(
                str(linear.x),
                str(linear.y),
                str(linear.z),
                str(angular.x),
                str(angular.y),
                str(angular.z),
            )
        )
        terra_msg = self.create_terrasentia_message(linear, angular)
        try:
            self.socket.sendall(bytes(terra_msg, "utf-8"))
        except Exception as e:
            self.get_logger().warn(
                "[TerraComm], FAILED to send message to socket. MSG = " + str(e)
            )

    def create_socket(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.connect((self.host, self.port))
        except Exception as e:
            self.get_logger().warn(
                "[TerraComm], FAILED to connect to socket, will continue anyways. MSG = "
                + str(e)
            )
        # TODO: some assertion to verify socket works
        return s

    # Creates message that gets sent to the terrasentia
    # This code is largely from Arun's code
    def create_terrasentia_message(self, linear, angular):
        header = "$CMD,"
        data_msg = (
            ","
            + str(time.time())
            + ","
            + str(linear.x)
            + ","
            + str(linear.y)
            + ","
            + str(linear.z)
            + ","
            + str(angular.x)
            + ","
            + str(angular.y)
            + ","
            + str(angular.z)
        )
        bytes_count = len(header) + len(data_msg)
        message = header + str(bytes_count + len(str(bytes_count))) + data_msg
        assert isinstance(message, str)
        return message


def twist_main(args=None):
    rclpy.init(args=args)

    terraComm = TerraComm()

    rclpy.spin(terraComm)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    terraComm.destroy_node()
    rclpy.shutdown()
