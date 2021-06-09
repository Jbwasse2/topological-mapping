#!/usr/bin/env python3
# This node takes the model trianed in the similarity directory in order to facilitate
# similarity detection for building the topological map
import torch
from torch import nn
import torchvision.transforms as transforms
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
import rclpy
from data.indoorData.results.similarity.best_model.model import Siamese

import pudb


class SimilarityService(Node):
    def __init__(self):
        super().__init__("similarity_publisher")
        self.results = []
        self.srv = self.create_service(
            Similarity, "similarity", self.get_similarity)
        self.model = self.get_model()

    def get_model(self):
        model = Siamese()
        weight_path = "./data/indoorData/results/similarity/best_model/saved_model.pth"
        model.load_state_dict(torch.load(weight_path))
        model.eval()
        print("FINISHED LOADING MODEL")
        return model

    # This expects the data images to come in as "cleaned rgb8" images
    # So the image should not be raw image from TerraSentia

    def get_similarity(self, request, response):
        def prepare_data(image):
            image = cv2.resize(image, (224, 224)) / 255
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            image = transform(image)
            return image.unsqueeze(0).float()
        bridge = CvBridge()
        image1 = bridge.imgmsg_to_cv2(request.image1, "bgr8")
        image2 = bridge.imgmsg_to_cv2(request.image2, "bgr8")
        image1 = prepare_data(image1)
        image2 = prepare_data(image2)
        results = self.model(image1, image2)
        prob = nn.functional.softmax(results)
        positive_prob = prob[0][1].cpu().detach()
        response.results = True if positive_prob > request.confidence else False
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
