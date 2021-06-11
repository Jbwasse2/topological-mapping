import networkx as nx
import torch
import torch.nn.functional as F
from rclpy.node import Node
import rclpy
from rclpy.qos import QoSProfile
from top_map_msg_srv.srv import Similarity
from std_msgs.msg import Header
from cv_bridge import CvBridge
from data.indoorData.results.similarity.best_model.model import Siamese
from sensor_msgs.msg import Image
from torch import nn


class EmbeddingGetter(Siamese):
    def __init__(self):
        super().__init__()
        model = self.get_model()

    def get_model(self):
        model = Siamese()
        weight_path = "./data/indoorData/results/similarity/best_model/saved_model.pth"
        model.load_state_dict(torch.load(weight_path))
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


# Class for building topological map


class TopologicalMap(Node):
    # Distance is in meters
    def __init__(self, use_pose_estimate=True, close_distance=1, confidence=0.8):
        super().__init__("Topological_Map")
        self.map = nx.DiGraph()
        self.embeddingGetter = EmbeddingGetter()
        self.embeddingClassifier = EmbeddingsClassifier()
        self.use_pose_estimate = use_pose_estimate
        self.confidence = confidence
        q = QoSProfile(history=2)
        self.subscription = self.create_subscription(
            Image,
            "/terrasentia/usb_cam_node/image_raw",
            self.image_callback,
            qos_profile=q,
        )
        self.bridge = CvBridge()
        self.cli = self.create_client(Similarity, "similarity")
        while not self.cli.wait_for_service(timeout_sec=0.5):
            self.get_logger().info("service not available, waiting again...")
        self.req = Similarity.Request()
        self.counter = 0
        self.embedding_dict = {}

    def fix_camera_image(self, image):
        image = np.flipud(image)
        image = np.fliplr(image)
        return image

    def image_callback(self, msg):
        # If use_pose_estimate determine if any images close by
        pu.db
        image1 = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        image1 = self.fix_camera_image(image1)
        if self.use_pose_estimate:
            pass
        else:
            # If not using use_pose_estimate compare to every other image

            pass
        self.send_request(image1, image2)
        # Wait for feedback...

    def send_request(self, frame1, frame2):
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
        self.req.confidence = self.confidence
        self.future = self.cli.call_async(self.req)
