from top_map.camera import CameraPublisher
from multiprocessing import Process
import signal
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from gtesting_helper import run_node


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