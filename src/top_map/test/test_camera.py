import subprocess
from multiprocessing import Process

import cv2
from top_map.util import play_rosbag

import rclpy
from rclpy.task import Future
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class BagTester(Node):
    def __init__(self, timeout=None):
        super().__init__("bag_tester")
        self.future = future
        self.results = []
        self.subscription = self.create_subscription(
            Image, "/terrasentia/usb_cam_node/image_raw", self.image_callback2, 1
        )
        self.bridge = CvBridge()
        self.image_number = 0
        self.results = False
        if timeout is not None:
            self.timer = self.create_timer(timeout, self.timer_callback)

    def timer_callback(self):
        self.future.set_result("Timeout")

    def image_callback2(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        cv2.imwrite(
            "./test/results/bag/camera" + str(self.image_number) + ".png", image
        )
        self.image_number += 1
        self.future.set_result("Pass")


def test_bag():
    rclpy.init()
    rosbag_location = "./test/testing_resources/rosbag/test_short.bag"
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
    bag_tester = BagTester(future, timeout=5.0)
    rclpy.spin_until_future_complete(bag_tester, future)
    bag_tester.destroy_node()
    # kills test.bag
    kill_testbag_cmd = (
        ". /opt/ros/melodic/setup.sh && rosnode list "
        + "| grep play | xargs rosnode kill"
    )
    subprocess.Popen(
        kill_testbag_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        shell=True,
    )
    rclpy.shutdown()
    assert future.result() == "Pass"


if __name__ == "__main__":
    test_bag()
