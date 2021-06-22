import networkx as nx
import time
import cv2
import pickle
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile
from top_map.similarityService import SimilarityService
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry


# Class for building topological map
class TopologicalMap(Node):
    # Distance is in meters

    def __init__(
        self,
        use_pose_estimate=False,
        close_distance=1,
        confidence=0.85,
        save_meng=True,
    ):
        super().__init__("Topological_Map")
        self.map = nx.DiGraph()
        self.similarityService = SimilarityService()
        self.use_pose_estimate = use_pose_estimate
        self.confidence = confidence
        self.close_distance = close_distance
        self.save_meng = save_meng
        q = QoSProfile(history=0)
        self.subscription = self.create_subscription(
            Image,
            "/terrasentia/usb_cam_node/image_raw",
            self.image_callback,
            qos_profile=q,
        )
        self.position = None
        self.orientation = None
        self.subscription_ekf = self.create_subscription(
            Odometry,
            "/terrasentia/ekf",
            self.ekf_callback,
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
        self.ekf_pose = {}
        self.debug_counter = 0

    def ekf_callback(self, msg):
        self.position = msg.pose.pose.position
        self.get_logger().info("Position is now " + str(self.position))
        self.orientation = msg.pose.pose.orientation

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
            "ekf_pose": self.ekf_pose,
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
        self.ekf_pose = info["ekf_pose"]

    def fix_camera_image(self, image):
        image = np.flipud(image)
        image = np.fliplr(image)
        return image

    # Input
    # reference_position: position of reference node
    # pose_dict: dictionary of poses for nodes in graph
    # top_map: topological map to search over for close nodes
    # threshold_distance: if below the distance threshold, the nodes
    # will be check with a deep learning model
    # output
    # All nodes that are close via pose estimate
    def get_all_close_nodes(
        self, reference_position, pose_dict, top_map, threshold_distance
    ):
        ret = []
        for node in top_map.nodes:
            node_position = pose_dict[node]["position"]
            distance = np.sqrt(
                (node_position.x - reference_position.x) ** 2
                + (node_position.y - reference_position.y) ** 2
                + (node_position.z - reference_position.z) ** 2
            )
            if distance < threshold_distance:
                ret.append(node)
        return ret

    def image_callback(self, msg):
        self.debug_counter += 1
        # If use_pose_estimate determine if any images close by
        image1 = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        image1 = self.fix_camera_image(image1)
        image1_embedding = self.similarityService.get_embedding(image1)
        image1_label = self.counter
        image1_trajectory_label = self.trajectory_label
        self.current_node = (image1_trajectory_label, image1_label)
        if self.use_pose_estimate:
            if self.position is not None:
                start = time.time()
                close_nodes = self.get_all_close_nodes(
                    self.position, self.ekf_pose, self.map, self.close_distance
                )
                self.get_logger().info(
                    "Time to run close_nodes=" + str(time.time() - start)
                )
                start = time.time()
                self.update_map(close_nodes, image1_embedding, image1)
                self.get_logger().info(
                    "Time to run update map=" + str(time.time() - start)
                )
        else:
            start = time.time()
            self.update_map(self.map.nodes, image1_embedding, image1)
            self.get_logger().info("Time to run update map=" + str(time.time() - start))

    def update_map(self, nodes_to_it_over, image1_embedding, image1):
        # If not using use_pose_estimate compare to every other image
        # If something is "similar" to image1, then don't add it.
        for node in nodes_to_it_over:
            image2_embedding = self.embedding_dict[node]
            closeness_indicator = self.similarityService.get_similarity_embedding(
                image1_embedding, image2_embedding, self.confidence
            )
            if closeness_indicator:
                self.current_node = node
                self.get_logger().info("Embeddings are close...")
                break
        # If current_node already in map, this does nothing
        self.map.add_node(self.current_node)
        # DOES NOT UPDATE EMBEDDING INFORMATION IF ALREADY IN THERE!
        # MAY BE USEFUL TO EXPERIMENT WITH THIS
        if self.current_node not in self.embedding_dict:
            self.embedding_dict[self.current_node] = image1_embedding
            self.meng[self.current_node] = cv2.resize(image1, (64, 64))
            pose = {"position": self.position, "orientation": self.orientation}
            self.ekf_pose[self.current_node] = pose
            self.counter += 1
        if self.last_node is not None:
            if self.last_node != self.current_node:
                self.map.add_edge(self.last_node, self.current_node)
        self.last_node = self.current_node
        self.current_node = None

    def loop_closure(self, graph):
        for node in self.map.nodes:
            node_position = self.ekf_pose[node]["position"]
            close_nodes = self.get_all_close_nodes(
                node_position, self.ekf_pose, self.map, self.close_distance
            )
            for neighboring_node in close_nodes:
                if neighboring_node == node:
                    continue
                image1_embedding = self.embedding_dict[node]
                image2_embedding = self.embedding_dict[neighboring_node]
                closeness_indicator = self.similarityService.get_similarity_embedding(
                    image1_embedding, image2_embedding, self.confidence
                )
                if closeness_indicator:
                    self.get_logger().info(
                        "Mergin nodes " + str(node) + " and " + str(neighboring_node)
                    )
                    # Move all edges from neighboring node to node
                    # First incoming edges
                    incoming_edges = graph.in_edges(neighboring_node)
                    for edge in list(incoming_edges):
                        other_node = edge[0]
                        if other_node != node:
                            graph.edges((other_node, node))
                    # Next outgoing edges
                    outgoing_edges = graph.out_edges(neighboring_node)
                    for edge in list(outgoing_edges):
                        other_node = edge[1]
                        if other_node != node:
                            graph.edges((node, other_node))
                    # Remove neighoring node from map
                    graph.remove_edge(neighboring_node)
