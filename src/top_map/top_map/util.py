import os
import subprocess
from multiprocessing import Process
import time

import rclpy


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
def play_rosbag(bag_location, loop=False, rosbag_args=None):
    assert os.path.exists(bag_location)
    while 1:
        # There is a mixing problem with just calling the rosbag
        # with py2.7 and py3.6 that occurs due to sourcing both
        # ros1 and ros2 respectively. This circumvents this sourcing problem
        s1 = "export PYTHONPATH='' && "
        s2 = ". /opt/ros/melodic/setup.sh && "
        s3 = "printenv | grep PYTHONPATH && "
        s4 = "rosbag play -q " + bag_location + " " + str(rosbag_args)
        s = s1 + s2 + s3 + s4
        subprocess.Popen(
            s, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, shell=True
        )
        if loop is False:
            break
    # Keeps process from returning to foreground
    # Useful for when you are debugging with pudb
    # Otherwise the process would return to the foreground
    # Meanning you have to run fg to return to pudb


# In testing it is desireable to wait for the ROS2 nodes to die
# Unlike ROS1 where you can just kill (yeet) the nodes away,
# ROS2 makes you wait.
# Input: allowed_nodes : Nodes that we won't wait to be killed
#        timeout : int, maximum number of seconds to wait. If None, will
#                  not timeout.
# Returns whenever all of the nodes are dead (besides allowed_nodes) or timeout
def wait_for_ros2_nodes_to_die(allowed_nodes=["/ros_bridge"], timeout=None):
    if timeout is None:
        timeout = -1
    assert isinstance(timeout, int)
    while timeout != 0:
        command1 = "export PYTHONPATH='' && "
        command2 = ". /opt/ros/dashing/setup.sh && "
        command3 = "ros2 node list"
        command = command1 + command2 + command3
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
        )
        out, err = process.communicate()
        out = out.splitlines()
        # Checks if nodes that are still alive are in allowed_nodes
        flag = 1
        for node in out:
            string_node = node.decode("utf-8")
            if string_node not in allowed_nodes:
                flag = 0
        if flag:
            break
        timeout -= 1
        time.sleep(1)
    return


def kill_ros2_node(node):

    assert isinstance(node, str)
    command1 = "export PYTHONPATH='' && "
    command2 = ". /opt/ros/dashing/setup.sh && "
    command3 = "ros2 lifecycle set " + node + " shutdown"
    command = command1 + command2 + command3
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    out, err = process.communicate()
    return out


if __name__ == "__main__":
    #    wait_for_ros2_nodes_to_die()
    kill_ros2_node("similarity_service")
