from top_map.similarityService import SimilarityService
import os
import signal
from multiprocessing import Process
from top_map.util import run_node
from top_map_msg_srv.srv import Similarity
import rclpy
import numpy as np
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class SimilarityClient(Node):
    def __init__(self):
        super().__init__("similarity_client")
        self.bridge = CvBridge()
        self.cli = self.create_client(Similarity, "similarity")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("service not available, waiting again...")
        self.req = Similarity.Request()

    # sensor_msgs/Image image1
    # sensor_msgs/Image image2
    # float64 confidence
    def send_request(self):
        image1 = Image()
        image1.height = 280
        image1.width = 320
        image1.encoding = "rgb8"
        frame = self.create_random_image()
        value = self.bridge.cv2_to_imgmsg(frame.astype(np.uint8))
        image1.data = value.data
        image2 = Image()
        image2.height = 280
        image2.width = 320
        image2.encoding = "rgb8"
        frame = self.create_random_image()
        value = self.bridge.cv2_to_imgmsg(frame.astype(np.uint8))
        image2.data = value.data
        self.req.image1 = image1
        self.req.image2 = image2
        self.req.confidence = 0.5
        self.future = self.cli.call_async(self.req)

    def create_random_image(self):
        return np.random.randint(255, size=(280, 320, 3))


def test_Similarity():
    rclpy.init()
    args = {}
    p = Process(
        target=run_node,
        args=(SimilarityService, args),
    )
    results = 0
    p.start()
    minimal_client = SimilarityClient()
    minimal_client.send_request()
    while rclpy.ok():
        rclpy.spin_once(minimal_client)
        if minimal_client.future.done():
            try:
                minimal_client.future.result()
                results = 1
            except Exception as e:
                minimal_client.get_logger().info("Service call failed %r" % (e,))
            break

    os.kill(p.pid, signal.SIGKILL)
    minimal_client.destroy_node()
    rclpy.shutdown()
    assert results == 1
