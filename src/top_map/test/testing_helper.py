import os

import rclpy


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
    while 1:
        os.system("ros2 bag play " + bag_location)
        if loop is False:
            break


if __name__ == "__main__":
    play_rosbag("./rosbag/rosbag2_2021_04_14-09_01_00", True)
