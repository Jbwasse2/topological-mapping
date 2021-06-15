from top_map.similarityService import (
    EmbeddingGetter,
    EmbeddingsClassifier,
)
import torch
from top_map_msg_srv.srv import Similarity, EmbeddingSimilarity, GetEmbedding
import rclpy
import numpy as np
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


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
        timeout_count = 0
        max_timeout_count = 5
        if timeout is not None:
            self.timer = self.create_timer(timeout, self.timer_callback)
        self.bridge = CvBridge()
        self.cli1 = self.create_client(Similarity, "similarity_images")
        self.cli2 = self.create_client(GetEmbedding, "get_embedding")
        self.cli3 = self.create_client(EmbeddingSimilarity, "similarity_embeddings")
        while not self.cli1.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("service not available, waiting again...")
            timeout_count += 1
            assert timeout_count < max_timeout_count
        while not self.cli2.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("service not available, waiting again...")
            timeout_count += 1
            assert timeout_count < max_timeout_count
        while not self.cli3.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("service not available, waiting again...")
            timeout_count += 1
            assert timeout_count < max_timeout_count
        self.req_sim_image = Similarity.Request()
        self.req_get_embedding = GetEmbedding.Request()
        self.req_sim_embed = EmbeddingSimilarity.Request()
        self.results = []

    # Possible race condition if this is set before send_request goes off...
    def timer_callback(self):
        self.future.set_result("Timeout")

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
        self.req_sim_image.image1 = image1
        self.req_sim_image.image2 = image2
        self.req_sim_image.confidence = 0.5
        self.future = self.cli1.call_async(self.req_sim_image)

    # std_msgs/Header header
    # sensor_msgs/Image image
    # ---
    # std_msgs/Header header
    # float32[] embedding
    def send_request2(self):
        image1 = Image()
        image1.height = 280
        image1.width = 320
        image1.encoding = "rgb8"
        frame = self.create_random_image()
        value = self.bridge.cv2_to_imgmsg(frame.astype(np.uint8))
        image1.data = value.data
        self.req_get_embedding.image = image1
        self.future = self.cli2.call_async(self.req_get_embedding)

    # std_msgs/Header header
    # float32[] embedding1
    # float32[] embedding2
    # float64 confidence
    # ---
    # std_msgs/Header header
    # bool results
    def send_request3(self):
        self.req_sim_embed.embedding1 = self.create_random_embedding()
        self.req_sim_embed.embedding2 = self.create_random_embedding()
        self.req_sim_embed.confidence = 0.5
        self.future = self.cli3.call_async(self.req_sim_embed)

    def create_random_image(self):
        return np.random.randint(255, size=(280, 320, 3))

    def create_random_embedding(self):
        return np.random.rand(1, 512, 7, 7).astype("float32").flatten().tolist()


#
def test_Similarity():

    try:
        #        args = {}
        #        p = Process(
        #            target=run_node,
        #            args=(SimilarityService, args),
        #        )
        #        p.start()
        results = [0, 0, 0]
        minimal_client = SimilarityClient(timeout=10)
        minimal_client.send_request()
        while rclpy.ok():
            rclpy.spin_once(minimal_client)
            if minimal_client.future.done():
                try:
                    test_results = minimal_client.future.result().results
                    if test_results != "Timeout" and (
                        test_results is True or test_results is False
                    ):
                        results[0] = 1
                except Exception as e:
                    minimal_client.get_logger().info("Service call failed %r" % (e,))
                break
        minimal_client.send_request2()
        while rclpy.ok():
            rclpy.spin_once(minimal_client)
            if minimal_client.future.done():
                try:
                    test_results = minimal_client.future.result()
                    if (
                        test_results != "Timeout"
                        and len(test_results.embedding) == 512 * 7 * 7
                    ):
                        results[1] = 1
                except Exception as e:
                    minimal_client.get_logger().info("Service call failed %r" % (e,))
                break
        minimal_client.send_request3()
        while rclpy.ok():
            rclpy.spin_once(minimal_client)
            if minimal_client.future.done():
                try:
                    test_results = minimal_client.future.result().results
                    if test_results != "Timeout" and (
                        test_results is True or test_results is False
                    ):
                        results[2] = 1
                except Exception as e:
                    minimal_client.get_logger().info("Service call failed %r" % (e,))
                break
        # Run other tests, if they are run as their own all the tests break for some reason...
        atest_get_classification()
        atest_get_embedding()
    except Exception as e:
        raise e
    else:
        pass
    finally:
        minimal_client.destroy_subscription(minimal_client.cli1)
        minimal_client.destroy_subscription(minimal_client.cli2)
        minimal_client.destroy_subscription(minimal_client.cli3)
        #        os.kill(p.pid, signal.SIGKILL)
        minimal_client.destroy_node()
        assert results[0] == 1
        assert results[1] == 1
        assert results[2] == 1
        assert minimal_client.future.result() != "Timeout"
