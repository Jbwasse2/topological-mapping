import os
from multiprocessing import Process

import rclpy
from rclpy.node import Node


# Some of the tests and even some operational code needs to run a rosbag
# after initializing, this class wraps around a given Node class
# and initializes it in the usual way then calls play_rosbag
# This function makes a class that inherits from the input "wrap_node"
# This function returns the CLASS NOT AN INSTANCE! 
# This is because I already wrote run_node to take a class, not an instance.
def bag_wrapper(wrap_node, rosbag_location, kwargs):
    class BagWrapper(wrap_node):
        def __init__(self, wrap_node, rosbag_location, kwargs):
            super().__init__(**kwargs)
            p = Process(
                target=play_rosbag,
                args=(rosbag_location, False),
            )
            p.start()
    return BagWrapper


# This function is mostly used to run nodes in the background to test
# the functionality of other nodes. 
# Input : node_type : class (not instance) of node that we'd like to run
#         args : Any associated arguments for the node class in dictionary form
#                for example if the argument for a node's init is "foo" and "bar"
#                args = {"foo": 1, "bar": "justin is cool"}
def run_node(node_type, args):
    # Get rid of ROS messages from appearing during testing
    rclpy.logging._root_logger.set_level(50)
    node_instance = node_type(**args)
    rclpy.spin(node_instance)
    node_instance.destroy_node()


# Input: bag_location : directory to ROS2 bag location
#       loop : Should the rosbag be played indefinetly?
def play_rosbag(bag_location, loop=False):
    assert os.path.exists(bag_location)
    while 1:
        os.system("ros2 bag play " + bag_location)
        if loop is False:
            break


if __name__ == "__main__":
    play_rosbag("./testing_resources/rosbag/rosbag2_2021_04_14-09_01_00", False)
