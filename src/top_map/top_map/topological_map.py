import pickle
import rclpy
import os
import time
from copy import deepcopy

import cv2
import networkx as nx
import numpy as np
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image

from top_map.similarityService import SimilarityService


# Class for building topological map
class TopologicalMap(Node):
    # Distance is in meters

    def __init__(self):
        super().__init__("Topological_Map")
        self.get_logger().info("Topological Map is Starting")
        self.declare_parameters(
            namespace="",
            parameters=[
                ("use_pose_estimate", True),
                ("close_distance", 1.0),
                ("confidence", 0.85),
                ("save_meng", True),
            ],
        )
        use_pose_estimate = self.get_parameter("use_pose_estimate").value
        close_distance = self.get_parameter("close_distance").value
        confidence = self.get_parameter("confidence").value
        save_meng = self.get_parameter("save_meng").value
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
        # Meng requires 5 before images and 5 images after goal
        self.BUFFER_SIZE = 5
        self.meng_buffer = [None for _ in range(self.BUFFER_SIZE)]
        # meng buffer after keeps track of images that comes up
        # after an image is added to the map
        self.meng_buffer_after = {}
        self.ekf_pose = {}
        self.debug_counter = 0
        self.get_logger().info("Topological Map is Ready")

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
        assert os.path.exists(location)
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

    # image : image given at a given time
    # meng_after_dict : dict of keys=node and values=list
    # where list keep track of images for saving for meng after
    # goal has been added to topological map
    def update_meng_after_dict(self, image, meng_after_dict):
        # Add image to all lists
        copy_of_meng_after_dict = deepcopy(meng_after_dict)
        for key, value in copy_of_meng_after_dict.items():
            value.pop(0)
            value.append(cv2.resize(image, (64, 64)))
            meng_after_dict[key] = value
            # If list is now full, update self.meng
            if not any(elem is None for elem in value):
                self.meng[key] += value
                meng_after_dict.pop(key, None)
        return meng_after_dict

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
        # Add image to buffer so that way it can be saved later
        self.meng_buffer.pop(0)
        self.meng_buffer.append(cv2.resize(image1, (64, 64)))
        self.meng_buffer_after = self.update_meng_after_dict(
            image1, self.meng_buffer_after
        )
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
            self.meng[self.current_node] = self.meng_buffer + [
                cv2.resize(image1, (64, 64))
            ]
            self.meng_buffer_after[self.current_node] = [
                None for i in range(self.BUFFER_SIZE)
            ]
            pose = {"position": self.position, "orientation": self.orientation}
            self.ekf_pose[self.current_node] = pose
            self.counter += 1
        if self.last_node is not None:
            if self.last_node != self.current_node:
                self.map.add_edge(self.last_node, self.current_node)
        self.last_node = self.current_node
        self.current_node = None

    def loop_closure(self, graph):
        nodes_to_it_over = deepcopy(self.map.nodes)
        for node in nodes_to_it_over:
            # Node may have already been removed from graph, if it is then
            # skip over it
            if node not in self.map.nodes:
                continue
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
                    graph.remove_node(neighboring_node)


def main(args=None):
    rclpy.init(args=args)

    # Video stream doesnt work when ssh into machine and then run docker. TODO
    node = TopologicalMap()
    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
