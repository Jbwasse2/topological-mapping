# Planner should be to localize and plan over the topological map
from operator import itemgetter
import rclpy
from top_map.topological_map import TopologicalMap
import random

import networkx as nx
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image


class Planner(Node):
    # If update rate is none, the planner will localize once and then
    # create a trajectory based on predicted localizations
    # Otherwise if not None, the robot will update the path at a rate given.
    def __init__(self, update_rate=None, confidence=0.85, seed=0):
        random.seed(seed)
        super().__init__("Planner")
        self.declare_parameter(
            "topological_map_pkl", "./test/testing_resources/test_top_map.pkl"
        )
        topological_map_pkl = self.get_parameter("topological_map_pkl").value
        self.subscription = self.create_subscription(
            Image, "/terrasentia/usb_cam_node/image_raw", self.image_callback
        )
        self.publisher = self.create_publisher(
            Image,
            "/top_map/local_goal",
            2,
        )
        self.top_map = TopologicalMap()
        self.top_map.load(topological_map_pkl)
        # Tries to set a good goal
        self.local_goal = None
        self.goal = self.get_random_node(self.top_map.map.nodes)
        self.current_node = None
        self.plan = None
        self.confidence = confidence
        self.get_logger().info("Planner is ready")

    # Chooses a random node in the map as a goal
    def get_random_node(self, nodes):
        nodes = list(nodes)
        return random.choice(nodes)

    def plan_path(self, graph, start, goal):
        path = nx.algorithms.shortest_paths.weighted.dijkstra_path(graph, start, goal)
        assert len(path) > 0
        return path

    def image_callback(self, msg):
        image1 = self.top_map.bridge.imgmsg_to_cv2(msg, "rgb8")
        image1 = self.top_map.fix_camera_image(image1)
        image1_embedding = self.top_map.similarityService.get_embedding(image1)
        if self.current_node is None:
            self.localize(image1_embedding)
        else:
            local_goal_embedding = self.top_map.embedding_dict[self.local_goal]
            closeness_indicator = (
                self.top_map.similarityService.get_similarity_embedding(
                    image1_embedding, local_goal_embedding, self.confidence
                )
            )
            if closeness_indicator:
                # Advance local goal to next node in path
                self.get_logger().info("Updating Robot Location!")
                index = self.plan.index(self.local_goal)
                if index + 1 >= len(self.plan):
                    self.get_logger().warn(
                        "Tried updating plan, but no more path left!"
                    )
                else:
                    self.local_goal = self.plan[index + 1]
                    self.set_local_goal(self.top_map.meng[self.local_goal])

    # This is used for meng code in order to suggest waypoints
    # Meng image is saved as RGB8, but it actually needs to be sent out as BGR8
    def set_local_goal(self, image):
        image_msg = Image()
        image_msg.height = image[0].shape[0] * 11
        image_msg.width = image[0].shape[1]
        image_msg.encoding = "bgr8"
        image = np.vstack(image)
        value = self.top_map.bridge.cv2_to_imgmsg(image.astype(np.uint8))
        image_msg.data = value.data
        self.publisher.publish(image_msg)

    def localize(self, image1_embedding):
        nodes_to_it_over = self.top_map.map.nodes
        close_nodes = []
        for node in nodes_to_it_over:
            image2_embedding = self.top_map.embedding_dict[node]
            closeness_indicator, prob = (
                self.top_map.similarityService.get_similarity_embedding(
                    image1_embedding, image2_embedding, self.confidence
                )
            )
            if closeness_indicator:
                close_nodes.append((node, prob))
        if len(close_nodes) == 1:
            self.current_node = close_nodes[0][0]
            self.plan = self.plan_path(
                self.top_map.map, self.current_node, self.goal
            )
            self.local_goal = self.plan[1]
            self.set_local_goal(self.top_map.meng[self.local_goal])
            self.get_logger().info("Localized Robot in Map!")
        elif len(close_nodes) == 0:
            self.get_logger().warning("Failed to Localize Robot in Map!")
        else:
            self.get_logger().warning("Multiple Possible Local Nodes Detected!")
            self.get_logger().warning(str(close_nodes))
            #Choose node with highest prob of similarity
            best_node = max(close_nodes, key=itemgetter(1))
            self.current_node = best_node[0]
            self.plan = self.plan_path(
                self.top_map.map, self.current_node, self.goal
            )
            self.local_goal = self.plan[1]
            self.set_local_goal(self.top_map.meng[self.local_goal])


def main(args=None):
    rclpy.init(args=args)

    # Video stream doesnt work when ssh into machine and then run docker. TODO
    node = Planner()
    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
