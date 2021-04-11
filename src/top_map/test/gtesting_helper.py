import signal
import os
from multiprocessing import Process
from rclpy.node import Node
import pudb  # noqa
import rclpy
import sys


def run_node(node_type, args):
    # Get rid of ROS messages from appearing during testing
    rclpy.logging._root_logger.set_level(50)
    node_instance = node_type(**args)
    rclpy.spin(node_instance)
    node_instance.destroy_node()
