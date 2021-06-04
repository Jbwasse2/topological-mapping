# This file iterates over the data directory and finds all of the bag files
# This script will then get the pose of the robot in the frames with the following
# For each rosbag
#   restart orbslam
#   play rosbag
#   Collect robot pose for all frames
#   save poses to file named similar to rosbag in same location of rosbag
import glob
import os
import pickle
from natsort import natsorted
import signal
import time
from multiprocessing import Process
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pudb
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.task import Future
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from top_map.pose import Orbslam2Pose
from top_map.util import bag_wrapper, play_rosbag, run_node


class PoseWriter(Node):
    def __init__(self, future):
        super().__init__("pose_tester")
        self.future = future
        # https://docs.ros2.org/latest/api/rclpy/api/qos.html#rclpy.qos.QoSProfile
        q = QoSProfile(history=2)
        self.subscription = self.create_subscription(
            PoseStamped, "pose", self.pose_callback, qos_profile=q
        )
        self.poses = []
        self.counter = 0
        # Bit jank, but need to check if rosbag finished, use a timer to check this.
        # Set up a flag to determine if rosbag/orbslam2 has started playing
        # if it has start the timer.
        self.set_timer = 0
        # If flag is 0, then its been a second since last message, so kill self
        self.flag = 1

    def timer_callback(self):
        if self.flag:
            self.flag = 0
        else:
            self.future.set_result("Timeout")
            self.get_logger().info("Timing Out!")
            return

    def pose_callback(self, msg):
        if self.set_timer == 0:
            self.timer = self.create_timer(1, self.timer_callback)
            self.get_logger().info("Starting pose callback timer")
            self.set_timer = 1
        self.get_logger().info("Got pose " + str(self.counter))
        self.poses.append(
            (
                msg.header.frame_id,
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
                msg.pose.orientation.w,
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
            )
        )
        self.flag = 1
        self.counter += 1


class ImagesToCamera(Node):
    def __init__(self, image_dir):
        super().__init__("image_to_camera")
        self.timer = self.create_timer(1 / 30, self.timer_callback)
        self.counter = 0
        self.images = self.get_images(image_dir)
        self.bridge = CvBridge()
        self.publisher_ = self.create_publisher(Image, "camera", 1)

    def get_images(self, image_dir):
        return list(natsorted(Path(image_dir).rglob("*.jpg")))

    def timer_callback(self):
        image = cv2.imread(str(self.images[self.counter]))
        msg = Image()
        header = Header()
        header.frame_id = str(self.counter)
        header.stamp = self.get_clock().now().to_msg()
        msg.header = header
        msg.height = image.shape[0]
        msg.width = image.shape[1]
        msg.encoding = "bgr8"
        value = self.bridge.cv2_to_imgmsg(image.astype(np.uint8))
        msg.data = value.data
        self.publisher_.publish(msg)
        self.counter += 1


def collect_data():
    rosbags = Path("./data/bags/").rglob("*.db3")
    for rosbag in rosbags:
        # In the future should check to make sure text file
        # doesn't already exist. Could probably also optimize this
        # by reseting orbslam node instead of making new instance
        pose_args = {"visualize": False, "slam_settings_path": "./configs/gibson.yaml"}
        p = Process(
            target=run_node,
            args=(
                Orbslam2Pose,
                pose_args,
            ),
        )
        p.start()
        # Give time for orbslam2 to init
        time.sleep(15)
        # It is easier to glob over bags for potential files, so just change string a
        # bit in order to get proper image location
        parent = str(rosbag.parent).replace("bags", "clean")
        parent = "./data/gibson/Browntown"
        image_args = {"image_dir": parent}
        p_images = Process(
            target=run_node,
            args=(
                ImagesToCamera,
                image_args,
            ),
        )
        p_images.start()
        future = Future()
        poseWriter = PoseWriter(future)
        rclpy.spin_until_future_complete(poseWriter, future)
        fp = open(str(parent) + ".pkl", "wb")
        pickle.dump(poseWriter.poses, fp)
        os.kill(p.pid, signal.SIGKILL)
        break


if __name__ == "__main__":
    rclpy.init()
    collect_data()
    rclpy.shutdown()

