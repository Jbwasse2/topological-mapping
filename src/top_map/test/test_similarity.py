import numpy as np
import torch
from rclpy.node import Node

from top_map.similarityService import EmbeddingGetter, EmbeddingsClassifier


def atest_get_embedding():
    embeddingGetter = EmbeddingGetter()
    random_image = np.random.randint(255, size=(1, 3, 224, 224))
    random_image = torch.from_numpy(random_image).float()
    embedding = embeddingGetter(random_image)
    assert embedding.shape == torch.Size([1, 512, 7, 7])


def atest_get_classification():
    embeddingClassifier = EmbeddingsClassifier().cpu()
    embedding1 = np.random.rand(1, 512, 7, 7)
    embedding2 = np.random.rand(1, 512, 7, 7)
    embedding1 = torch.from_numpy(embedding1).float()
    embedding2 = torch.from_numpy(embedding2).float()
    prediction = embeddingClassifier(embedding1, embedding2)
    assert prediction.shape == torch.Size([1, 2])


class SimilarityClient(Node):
    def __init__(self, timeout=None):
        super().__init__("similarity_client_test")
        self.results = []

    # Possible race condition if this is set before send_request goes off...
    def timer_callback(self):
        self.future.set_result("Timeout")


def test_Similarity():
    try:
        atest_get_classification()
        atest_get_embedding()
    except Exception as e:
        raise e
    else:
        pass
    finally:
        pass
