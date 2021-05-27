import glob
import os
import signal
from multiprocessing import Process

import cv2
from top_map.camera import CameraPublisher
from top_map.util import play_rosbag, run_node

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class CameraTester(Node):
    def __init__(self):
        super().__init__("camera_tester")
        self.results = []
        self.subscription = self.create_subscription(
            Image, "camera", self.image_callback2, 1
        )

    def image_callback2(self, msg):
        self.results.append(msg.height == 480)
        self.results.append(msg.width == 640)
        self.results.append(msg.encoding == "bgr8")
        # Check if image is empty, if it is camera isn't on
        # Also check if camera exists, if it doesn't, don't fail
        self.results.append(sum(msg.data) != 0 or len(
            glob.glob("/dev/video?")) == 0)
        bridge = CvBridge()
        image = bridge.imgmsg_to_cv2(msg, "bgr8")
        cv2.imwrite("./test/results/camera.png", image)


class BagTester(Node):
    def __init__(self, timeout=None):
        super().__init__("bag_tester")
        self.results = []
        self.subscription = self.create_subscription(
            Image, "camera", self.image_callback2, 1
        )
        self.bridge = CvBridge()
        self.image_number = 0
        self.results = False
        if timeout is not None:
            self.timer = self.create_timer(timeout, self.timer_callback)

    def timer_callback(self):
        self.destroy_node()

    def image_callback2(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        cv2.imwrite(
            "./test/results/bag/camera" +
            str(self.image_number) + ".png", image
        )
        self.image_number += 1
        self.results = True


def test_camera():
    rclpy.init()

    camera_args = {"camera_id": 1, "stream_video": False}
    p = Process(
        target=run_node,
        args=(
            CameraPublisher,
            camera_args,
        ),
    )
    p.start()
    results = run_test_camera()
    os.kill(p.pid, signal.SIGKILL)
    rclpy.shutdown()
    assert False not in results


def test_bag():
    rclpy.init()
    rosbag_location = "./test/testing_resources/rosbag/rosbag2_2021_04_14-09_01_00"
    p = Process(
        target=play_rosbag,
        args=(rosbag_location, False),
    )
    p.start()
    bag_tester = BagTester(timeout=5.0)
    rclpy.spin_once(bag_tester)
    try:
        bag_tester.destroy_node()
    # Already killed
    except rclpy.handle.InvalidHandle:
        pass
    os.kill(p.pid, signal.SIGKILL)
    rclpy.shutdown()
    assert bag_tester.results is not False


def run_test_camera():
    camera_tester = CameraTester()
    rclpy.spin_once(camera_tester)
    camera_tester.destroy_node()
    return camera_tester.results
