#!/usr/bin/env python3
# This node takes the model trianed in the similarity directory in order to facilitate
# similarity detection for building the topological map
import cv2
import rclpy
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn

from top_map.top_data.indoorData.results.similarity.best_model.model import \
    Siamese


class EmbeddingGetter(Siamese):
    def __init__(self):
        super().__init__()
        self.model = self.get_model()

    def get_model(self):
        model = Siamese()
        weight_path = "./data/indoorData/results/similarity/best_model/saved_model.pth"
        model.load_state_dict(
            torch.load(weight_path, map_location=torch.device("cpu")))
        model.eval()
        return model

    def forward(self, x1):
        self.model.encoder.eval()
        out1 = self.model.encoder(x1)
        return out1


class EmbeddingsClassifier(EmbeddingGetter):
    def __init__(self):
        super(EmbeddingsClassifier, self).__init__()

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        x = self.model.conv1(x)
        x = F.relu(x)
        x = self.model.conv2(x)
        x = F.relu(x)
        x = self.model.conv3(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.model.fc1(x)
        x = F.relu(x)
        x = self.model.dropout1(x)
        x = self.model.fc2(x)
        return x


class SimilarityService:
    def __init__(self):
        self.embeddingGetter = EmbeddingGetter()
        self.embeddingClassifier = EmbeddingsClassifier()
        self.model = self.get_model()

    def get_model(self):
        model = Siamese()
        weight_path = "./data/indoorData/results/similarity/best_model/saved_model.pth"
        model.load_state_dict(
            torch.load(weight_path, map_location=torch.device("cpu")))
        model.eval()
        return model

    def get_embedding(self, image):
        image = self.prepare_data(image)
        embedding = self.embeddingGetter(image).cpu().detach().numpy()
        return embedding

    # 1,512,7,7 torch float
    def get_similarity_embedding(self, embedding1, embedding2, confidence):
        embedding1 = torch.from_numpy(embedding1)
        embedding2 = torch.from_numpy(embedding2)
        results = self.embeddingClassifier(embedding1, embedding2)
        prob = nn.functional.softmax(results)
        positive_prob = prob[0][1].cpu().detach()
        response = True if positive_prob > confidence else False
        return response, positive_prob.tolist()

    # This expects the data images to come in as "cleaned rgb8" images
    # So the image should not be raw image from TerraSentia, but is that but
    # flipped horizontally, vertically, and made into RGB8
    def get_similarity(self, image1, image2, confidence):
        image1 = self.prepare_data(image1)
        image2 = self.prepare_data(image2)
        results = self.model(image1, image2)
        prob = nn.functional.softmax(results)
        positive_prob = prob[0][1].cpu().detach()
        response = True if positive_prob > confidence else False
        return response

    def prepare_data(self, image):
        image = cv2.resize(image, (224, 224)) / 255
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        image = transform(image)
        return image.unsqueeze(0).float()


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
