# Some notes of Meng's code
# His waypoint predictor takes images as 64x64 BGR images with values 0-1
import shutil
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist, TwistStamped, Vector3
from rclpy.node import Node
from sensor_msgs.msg import Image
from topological_nav.reachability import model_factory


class WaypointPublisher(Node):
    # debug is neccessary because testing assumes the subscription is
    # started from the beggining. But at run time we only want to start
    # sending commands when a local goal is given
    def __init__(self):
        super().__init__("waypoint_publisher")
        self.declare_parameter("create_graphic", "./test/results/wp/")
        create_graphic = self.get_parameter("create_graphic").value
        # If output file doesn't exist create it for iamges to be saved to instead of displayed.
        if create_graphic != "":
            shutil.rmtree(create_graphic)
            Path(create_graphic).mkdir(parents=True, exist_ok=True)
            self.counter = 0

        self.create_graphic = create_graphic
        self.model = self.get_model()
        self.subscription = self.create_subscription(
            Image,
            "/terrasentia/usb_cam_node/image_raw",
            self.image_callback,
            2,
        )
        self.subscription = self.create_subscription(
            Image,
            "/top_map/local_goal",
            self.local_goal_callback,
            2,
        )
        self.start_moving = False
        self.publisher_ = self.create_publisher(TwistStamped,
                                                "/terrasentia/cmd_vel", 1)
        self.bridge = CvBridge()
        self.get_logger().info("Created Waypoint Node")
        self.goal = None
        self.count = 0

    def get_model(self):
        model = model_factory.get(
            "model_12env_v2_future_pair_proximity_z0228")(device="cpu")
        return model

    def create_waypoint_message(self, waypoint, reachability_estimator):
        lin = Vector3()
        angular = Vector3()
        # item converts from numpy float type to python float type
        lin.x = float(waypoint[0].item()) / 4
        lin.y = 0.0
        # msg.z = reachability_estimator.item()
        lin.z = 0.0
        angular.x = 0.0
        angular.y = 0.0
        angular.z = float(waypoint[1].item()) / 1
        msg = TwistStamped()
        msg_twist = Twist()
        msg_twist.linear = lin
        msg_twist.angular = angular
        msg.twist = msg_twist
        return msg

    def local_goal_callback(self, msg):
        images = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        images = np.reshape(images, (11, 64, 64, 3))
        self.goal = [images[i, :, :] for i in range(images.shape[0])]
        self.goal_show = self.goal[5]
        if self.start_moving is False:
            self.start_moving = True
            self.subscription = self.create_subscription(
                Image,
                "/terrasentia/usb_cam_node/image_raw",
                self.image_callback,
                2,
            )

    def image_to_waypoint(self, msg):
        self.get_logger().info("I heard {0}".format(str(msg.header)))
        image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        image = cv2.flip(image, -1)
        waypoint, reachability_estimator = self.get_wp(image, self.goal)
        self.get_logger().info("Reachability Estimator is {0}".format(
            str(reachability_estimator)))
        msg = self.create_waypoint_message(waypoint, reachability_estimator)
        self.publisher_.publish(msg)
        # The arrow is strictly for visualization, not used for navigation
        if self.create_graphic != "":
            arrow_start = (32, 20)
            arrow_end = (32 + int(10 * -waypoint[1]),
                         20 + int(10 * -waypoint[0]))
            color = (0, 0, 255)  # Red
            thickness = 2
            # cv2 like BGR because they like eating glue
            image = cv2.resize(image, (64, 64))
            image = np.hstack((image, cv2.resize(self.goal_show, (64, 64))))
            image = cv2.arrowedLine(image, arrow_start, arrow_end, color,
                                    thickness)
            (height, width, _) = image.shape
            # Add 0's to front to make it easier for script to make into video
            # counter should not be larger than 6 digits (IE 999999)
            counter_string = str(self.counter).rjust(6, "0")
            # https://stackoverflow.com/questions/16615662/how-to-write-text-on-a-image-in-windows-using-python-opencv2
            cv2.putText(image,
                        str(reachability_estimator)[0:4], (0, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
            cv2.imwrite(
                self.create_graphic + "frame" + counter_string + ".png", image)
            self.counter += 1

    def image_callback(self, msg):
        if self.goal is None:
            self.get_logger().info("No local goal set!")
        else:
            self.image_to_waypoint(msg)

    def get_wp(self, ob, goal):
        follower = self.model["follower"]
        goal = self.cv2_to_model_im(goal)
        ob = self.cv2_to_model_im(ob)
        return (
            follower.motion_policy.predict_waypoint(ob, goal),
            follower.sparsifier.predict_reachability(ob, goal),
        )

    def show_img(self, img):
        matplotlib.use("TkAgg")
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 0)
        plt.imshow(img)
        plt.show()

    # Cv2 gives images in BGR, and from 0-255
    # We want RGB and from 0-1
    # Can also get list/ np array of images, this should be handled
    def cv2_to_model_im(self, im):
        im = np.asarray(im)
        assert len(im.shape) == 3 or len(im.shape) == 4
        if len(im.shape) == 3:
            # (64a,64b,3) -> (3, 64a, 64b)
            # So we switch (0,2) and then (1,2)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (64, 64))
            im = np.swapaxes(im, 0, 2)
            im = np.swapaxes(im, 1, 2)
            im = np.asarray(im)
            im = (im / 255).astype("float32")
            return im
        else:
            # (11,64a,64b,3) -> (11, 3, 64a, 64b)
            # So we switch (1,3) and then (2,3)
            out_im = np.zeros((11, 64, 64, 3))
            for i in range(im.shape[0]):
                im_temp = cv2.cvtColor(im[i], cv2.COLOR_BGR2RGB)
                out_im[i] = cv2.resize(im_temp, (64, 64))
            out_im = np.swapaxes(out_im, 1, 3)
            out_im = np.swapaxes(out_im, 2, 3)
            out_im = np.asarray(out_im)
            out_im = (out_im / 255).astype("float32")
            return out_im

    def __del__(self):
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)

    waypoint_publisher = WaypointPublisher()

    rclpy.spin(waypoint_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    waypoint_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
