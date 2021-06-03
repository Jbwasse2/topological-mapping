import os
import subprocess
import signal
from multiprocessing import Process

import numpy as np
import pytest
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.task import Future
from sensor_msgs.msg import Image
from top_map.pose import Orbslam2Pose
from top_map.util import bag_wrapper, play_rosbag, run_node


# Checks if any pose message is sent
class PoseTester(Node):
    def __init__(self, future):
        super().__init__("pose_tester")
        self.subscription = self.create_subscription(
            PoseStamped, "pose", self.pose_callback, 1
        )
        self.timer = self.create_timer(60, self.timer_callback)
        self.future = future

    def pose_callback(self, msg):
        self.future.set_result("Pass")

    def timer_callback(self):
        self.future.set_result("Timeout")
        self.get_logger().info("Timing Out!")


# Checks if a good pose message is sent eventually.


class PoseTesterValid(PoseTester):
    def __init__(self, future):
        super().__init__(future)

    def pose_callback(self, msg):
        # Pass only if results are not inf
        if msg.pose.position.x == np.inf:
            pass
        else:
            self.future.set_result("Pass")


class BufferTester(Node):
    def __init__(self, future):
        super().__init__("pose_tester")
        q = QoSProfile(history=2)
        self.subscription = self.create_subscription(
            PoseStamped, "pose", self.pose_callback, qos_profile=q
        )
        self.image_counter = 0
        self.subscription = self.create_subscription(
            Image,
            "/terrasentia/usb_cam_node/image_raw",
            self.image_callback,
            qos_profile=q,
        )
        self.timer = self.create_timer(50, self.timer_callback)
        self.future = future
        self.counter = 0

    def image_callback(self, msg):
        self.image_counter += 1
        print(self.image_counter)

    def pose_callback(self, msg):
        self.counter += 1
        print(self.counter)

    def timer_callback(self):
        self.future.set_result("Timeout")
        self.get_logger().info("Timing Out!")
        self.get_logger().info("Counter = " + str(self.counter))
        self.get_logger().info("Image Counter = " + str(self.image_counter))


# This makes sure the QoS selected ensures that all messages are kept


@pytest.mark.skip(reason="Incomplete")
def test_orbslam2_buffer():
    rclpy.init()
    # bag_wrapper(wrap_node, rosbag_location, kwargs):
    rosbag_location = "./test/testing_resources/rosbag/test.bag"
    pose_args = {"visualize": False}
    orbslam2PoseWrapped = bag_wrapper(Orbslam2Pose, rosbag_location, pose_args)
    args = {
        "wrap_node": Orbslam2Pose,
        "rosbag_location": rosbag_location,
        "kwargs": pose_args,
    }
    p = Process(
        target=run_node,
        args=(
            orbslam2PoseWrapped,
            args,
        ),
    )
    p.start()
    future = Future()
    bufferTester = BufferTester(future)
    rclpy.spin_until_future_complete(bufferTester, future)
    bufferTester.destroy_node()
    os.kill(p.pid, signal.SIGKILL)
    rclpy.shutdown()
    assert future.result() == "Pass"


# First Check to make sure any pose message gets published
# Then Check to make sure a "good" message gets published
# A good message is one that isn't inf in all poses.
def test_orbslam2_message():
    rclpy.init()
    rosbag_location = "./test/testing_resources/rosbag/test_long.bag"
    pose_args = {"visualize": False}
    p2 = Process(
        target=run_node,
        args=(
            Orbslam2Pose,
            pose_args,
        ),
    )
    p2.start()
    p = Process(
        target=play_rosbag,
        args=(
            rosbag_location,
            False,
            "-l --topics /terrasentia/usb_cam_node/image_raw",
        ),
    )
    p.start()
    future = Future()
    pose = PoseTesterValid(future)
    rclpy.spin_until_future_complete(pose, future)
    pose.destroy_node()
    kill_testbag_cmd = (
        ". /opt/ros/melodic/setup.sh && "
        + "rosnode list | grep play | xargs rosnode kill"
    )
    subprocess.Popen(
        kill_testbag_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        shell=True,
    )
    os.kill(p.pid, signal.SIGKILL)
    os.kill(p2.pid, signal.SIGKILL)
    rclpy.shutdown()
    assert future.result() == "Pass"
