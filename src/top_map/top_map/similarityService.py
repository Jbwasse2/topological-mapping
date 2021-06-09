#!/usr/bin/env python3
# This node takes the model trianed in the similarity directory in order to facilitate
# similarity detection for building the topological map
import cv2
import numpy as np
import quaternion
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from top_map_msg_srv.srv import Similarity
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image
from std_msgs.msg import Header, Bool

import pudb


class SimilarityService(Node):
    def __init__(self):
        super().__init__("similarity_publisher")
        self.results = []
        self.srv = self.create_service(Similarity, "similarity", self.get_similarity)

    def get_similarity(self, request, response):
        print("Confidence is " + str(request.confidence))
        response.results = True
        return response


def main(args=None):
    rclpy.init(args=args)

    # Video stream doesnt work when ssh into machine and then run docker. TODO
    bag_publisher = SimilarityService()
    rclpy.spin(bag_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    bag_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
