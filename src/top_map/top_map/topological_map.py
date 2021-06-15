import networkx as nx
import cv2
import pickle
import time
import numpy as np
from rclpy.node import Node
import rclpy
from rclpy.qos import QoSProfile
from top_map_msg_srv.srv import Similarity, GetEmbedding, EmbeddingSimilarity
from std_msgs.msg import Header
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class SimilarityClient(Node):
    def __init__(self, wait_for_service=True):
        super().__init__("similarity_client")
        retries = 0
        max_retries = 5
        self.bridge = CvBridge()
        self.cli1 = self.create_client(Similarity, "similarity_images")
        self.cli2 = self.create_client(GetEmbedding, "get_embedding")
        self.cli3 = self.create_client(EmbeddingSimilarity, "similarity_embeddings")
        if wait_for_service:
            while not self.cli1.wait_for_service(timeout_sec=1.0):
                retries += 1
                assert retries < max_retries
                self.get_logger().info(
                    "Similarity service not available, waiting again..."
                )
            while not self.cli2.wait_for_service(timeout_sec=1.0):
                retries += 1
                assert retries < max_retries
                self.get_logger().info(
                    "GetEmbedding service not available, waiting again..."
                )
            while not self.cli3.wait_for_service(timeout_sec=1.0):
                retries += 1
                assert retries < max_retries
                self.get_logger().info(
                    "EmbeddingSimilarity service not available, waiting again..."
                )
        self.req_sim_image = Similarity.Request()
        self.req_get_embedding = GetEmbedding.Request()
        self.req_sim_embed = EmbeddingSimilarity.Request()

    def send_request_embedding_similarity(self, embedding1, embedding2, confidence):
        self.req_sim_embed.embedding1 = embedding1.astype("float32").flatten().tolist()
        self.req_sim_embed.embedding2 = embedding2.astype("float32").flatten().tolist()
        self.req_sim_embed.confidence = confidence
        self.future = self.cli3.call_async(self.req_sim_embed)

    def send_request_get_embedding(self, frame):
        image1 = Image()
        image1.height = frame.shape[0]
        image1.width = frame.shape[1]
        image1.encoding = "rgb8"
        value = self.bridge.cv2_to_imgmsg(frame.astype(np.uint8))
        image1.data = value.data
        self.req_get_embedding.image = image1
        self.future = self.cli2.call_async(self.req_get_embedding)

    def send_request_image_similarity(self, frame1, frame2, confidence):
        header = Header()
        header.frame_id = str(self.counter)
        header.stamp = self.get_clock().now().to_msg()
        image1 = Image()
        image1.height = frame1.shape[0]
        image1.width = frame1.shape[1]
        image1.encoding = "rgb8"
        value = self.bridge.cv2_to_imgmsg(frame1.astype(np.uint8))
        image1.data = value.data
        image2 = Image()
        image2.height = frame2.shape[0]
        image2.width = frame2.shape[1]
        image2.encoding = "rgb8"
        value = self.bridge.cv2_to_imgmsg(frame2.astype(np.uint8))
        image2.data = value.data
        self.req.header = header
        self.req.image1 = image1
        self.req.image2 = image2
        self.req.confidence = confidence
        self.future = self.cli.call_async(self.req)


# Class for building topological map
class TopologicalMap(Node):
    # Distance is in meters

    def __init__(
        self,
        use_pose_estimate=False,
        close_distance=1,
        confidence=0.85,
        save_meng=True,
        wait_for_service=True,
    ):
        super().__init__("Topological_Map")
        self.map = nx.DiGraph()
        self.similarityClient = SimilarityClient(wait_for_service)
        self.use_pose_estimate = use_pose_estimate
        self.confidence = confidence
        self.close_distance = close_distance
        self.save_meng = save_meng
        q = QoSProfile(history=2)
        self.subscription = self.create_subscription(
            Image,
            "/terrasentia/usb_cam_node/image_raw",
            self.image_callback,
            qos_profile=q,
        )
        self.counter = 0
        # TODO Allow for collecting of multiple trajectories...
        self.trajectory_label = 0
        self.embedding_dict = {}
        self.bridge = CvBridge()
        self.current_node = None
        self.last_node = None
        self.meng = {}
        self.debug_counter = 0

    def save(self, location="./top_map.pkl"):
        f = open(location, "wb")
        info = {
            "meng": self.meng,
            "current_node": self.current_node,
            "last_node": self.last_node,
            "embedding_dict": self.embedding_dict,
            "use_pose_estimate": self.use_pose_estimate,
            "close_distance": self.close_distance,
            "confidence": self.confidence,
            "save_meng": self.save_meng,
            "map": self.map,
        }
        pickle.dump(info, f)

    def load(self, location="./top_map.pkl"):
        f = open(location, "rb")
        info = pickle.load(f)
        self.meng = info["meng"]
        self.current_node = info["current_node"]
        self.last_node = info["last_node"]
        self.embedding_dict = info["embedding_dict"]
        self.use_pose_estimate = info["use_pose_estimate"]
        self.close_distance = info["close_distance"]
        self.confidence = info["confidence"]
        self.save_meng = info["save_meng"]
        self.map = info["map"]

    def fix_camera_image(self, image):
        image = np.flipud(image)
        image = np.fliplr(image)
        return image

    def image_callback(self, msg):
        self.debug_counter += 1
        print(self.debug_counter)
        # If use_pose_estimate determine if any images close by
        image1 = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        image1 = self.fix_camera_image(image1)
        self.similarityClient.send_request_get_embedding(image1)
        rclpy.spin_once(self.similarityClient)
        assert self.similarityClient.future.done()
        while not self.similarityClient.future.done():
            time.sleep(0.1)
            self.get_logger().info("Waiting for Get Embedding")
        image1_embedding = self.similarityClient.future.result().embedding
        image1_embedding = np.array(image1_embedding).reshape(1, 512, 7, 7)
        image1_label = self.counter
        image1_trajectory_label = self.trajectory_label
        self.current_node = (image1_trajectory_label, image1_label)
        if self.use_pose_estimate:
            raise NotImplementedError
        else:
            # If not using use_pose_estimate compare to every other image
            # If something is "similar" to image1, then don't add it.
            for node in self.map.nodes:
                image2_embedding = self.embedding_dict[node]
                self.similarityClient.send_request_embedding_similarity(
                    image1_embedding, image2_embedding, self.confidence
                )
                # BUG: For some reason I have to spin twice...
                rclpy.spin_once(self.similarityClient)
                rclpy.spin_once(self.similarityClient)
                while not self.similarityClient.future.done():
                    time.sleep(0.1)
                    self.get_logger().info("Waiting for Embedding Similarity")
                closeness_indicator = self.similarityClient.future.result().results
                if closeness_indicator:
                    self.current_node = node
                    self.get_logger().info("Embeddings are close...")
                    break
            # If current_node already in map, this does nothing
            self.map.add_node(self.current_node)
            # DOES NOT UPDATE EMBEDDING IF ALREADY IN THERE!
            # MAY BE USEFUL TO EXPERIMENT WITH THIS
            if self.current_node not in self.embedding_dict:
                self.embedding_dict[self.current_node] = image1_embedding
                self.meng[self.current_node] = cv2.resize(image1, (64, 64))
                self.counter += 1
            if self.last_node is not None:
                self.map.add_edge(self.last_node, self.current_node)
            self.last_node = self.current_node
            self.current_node = None