#!/usr/bin/env python3
# This node takes the model trianed in the similarity directory in order to facilitate
# similarity detection for building the topological map
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
from cv_bridge import CvBridge
from top_map_msg_srv.srv import Similarity, EmbeddingSimilarity, GetEmbedding
from rclpy.node import Node
import rclpy
from data.indoorData.results.similarity.best_model.model import Siamese


class EmbeddingGetter(Siamese):
    def __init__(self):
        super().__init__()
        model = self.get_model()

    def get_model(self):
        model = Siamese()
        weight_path = "./data/indoorData/results/similarity/best_model/saved_model.pth"
        model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
        model.eval()
        return model

    def forward(self, x1):
        self.encoder.eval()
        out1 = self.encode(x1)
        return out1


class EmbeddingsClassifier(EmbeddingGetter):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x


class SimilarityService(Node):
    def __init__(self):
        super().__init__("similarity_publisher")
        self.embeddingGetter = EmbeddingGetter()
        self.embeddingClassifier = EmbeddingsClassifier()
        self.results = []
        self.srv_sim_images = self.create_service(Similarity, "similarity_images", self.get_similarity)
        self.srv_embed = self.create_service(GetEmbedding, "get_embedding", self.get_embedding)
        self.srv_embed_sim = self.create_service(EmbeddingSimilarity, "similarity_embeddings", self.get_similarity_embedding)
        self.model = self.get_model()

    def get_model(self):
        model = Siamese()
        weight_path = "./data/indoorData/results/similarity/best_model/saved_model.pth"
        model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
        model.eval()
        return model

    def get_embedding(self, request, response):
        bridge = CvBridge()
        image1 = bridge.imgmsg_to_cv2(request.image, "rgb8")
        image1 = self.prepare_data(image1)
        embedding = self.embeddingGetter(image1).cpu().detach().numpy()
        response.embedding = embedding.flatten().tolist()
        return response

    def get_similarity_embedding(self, request, response):
        embedding1 = np.array(request.embedding1).reshape(1,512,7,7)
        embedding1 = torch.from_numpy(embedding1).float()
        embedding2 = np.array(request.embedding2).reshape(1,512,7,7)
        embedding2 = torch.from_numpy(embedding2).float()
        results = self.embeddingClassifier(embedding1, embedding2)
        prob = nn.functional.softmax(results)
        positive_prob = prob[0][1].cpu().detach()
        response.results = True if positive_prob > request.confidence else False
        return response

    def prepare_data(self, image):
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


    # This expects the data images to come in as "cleaned rgb8" images
    # So the image should not be raw image from TerraSentia, but is that but
    # flipped horizontally, vertically, and made into RGB8
    def get_similarity(self, request, response):
        bridge = CvBridge()
        image1 = bridge.imgmsg_to_cv2(request.image1, "rgb8")
        image2 = bridge.imgmsg_to_cv2(request.image2, "rgb8")
        image1 = self.prepare_data(image1)
        image2 = self.prepare_data(image2)
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
