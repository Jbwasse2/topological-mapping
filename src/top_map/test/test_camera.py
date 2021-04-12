import os
import signal
from multiprocessing import Process

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from top_map.camera import CameraPublisher

from testing_helper import run_node


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
        self.results.append(sum(msg.data) != 0)
        bridge = CvBridge()
        image = bridge.imgmsg_to_cv2(msg, "bgr8")
        cv2.imwrite("./test/results/camera.png", image)


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


def run_test_camera():
    camera_tester = CameraTester()
    rclpy.spin_once(camera_tester)
    camera_tester.destroy_node()
    return camera_tester.results
