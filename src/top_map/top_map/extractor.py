# The purpose of this file is to extract info from the ROS bag in order to
# train the deep learning models.
import rclpy
import cv2
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class BagExtractor(Node):
    def __init__(self):
        super().__init__("Bag_Extractor")
        self.results = []
        self.subscription = self.create_subscription(
            Image, "/terrasentia/usb_cam_node/image_raw", self.image_callback2, 100
        )
        self.counter = 0

    def image_callback2(self, msg):
        assert msg.encoding == "rgb8"
        # Check if image is empty, if it is camera isn't on
        # Also check if camera exists, if it doesn't, don't fail
        bridge = CvBridge()
        name = str(self.counter).zfill(7)
        image = bridge.imgmsg_to_cv2(msg, "rgb8")
        image = cv2.flip(image, -1)
        # Even though the Terrasentia says the encoding is RGB8, it is actually BGR8
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(
            "../data/indoorData/unorganized/" + name + ".jpg", image
        )
        self.counter += 1


def main(args=None):
    rclpy.init(args=args)

    # Video stream doesnt work when ssh into machine and then run docker. TODO
    bag_publisher = BagExtractor()
    rclpy.spin(bag_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    bag_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
