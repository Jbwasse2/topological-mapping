from top_map.camera import CameraPublisher
import signal
import os
import rclpy
from multiprocessing import Process
from rclpy.node import Node
from sensor_msgs.msg import Image
import pudb  # noqa


class CameraTester(Node):
    def __init__(self):
        super().__init__("camera_tester")
        self.results = []
        self.subscription = self.create_subscription(
            Image, "camera", self.image_callback, 1
        )

    def image_callback(self, msg):
        self.results.append(msg.height == 480)
        self.results.append(msg.width == 640)
        self.results.append(msg.encoding == "bgr8")
        # Check if image is empty, if it is camera isn't on
        self.results.append(sum(msg.data) != 0)


def test_camera():
    rclpy.logging._root_logger.set_level(50)
    rclpy.init()

    thread = Process(target=run_camera)
    thread.start()

    results = run_test_camera()
    os.kill(thread.pid, signal.SIGKILL)
    rclpy.shutdown()
    assert False not in results


def run_test_camera():
    camera_tester = CameraTester()
    rclpy.spin_once(camera_tester)
    camera_tester.destroy_node()
    return camera_tester.results


def run_camera():
    camera_publisher = CameraPublisher(camera_id=1, stream_video=False)
    rclpy.spin(camera_publisher)

    camera_publisher.destroy_node()
