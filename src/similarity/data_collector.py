# This file iterates over the data directory and finds all of the bag files
# This script will then get the pose of the robot in the frames with the following
# For each rosbag
#   restart orbslam
#   play rosbag
#   Collect robot pose for all frames
#   save poses to file named similar to rosbag in same location of rosbag
import glob
from multiprocessing import Process
from pathlib import Path

import pudb
from top_map.pose import Orbslam2Pose
from top_map.util import play_rosbag, run_node

import rclpy
from geometry_msgs.msg import Pose
from rclpy.node import Node


class PoseWriter(Node):
    def __init__(self):
        super().__init__("pose_tester")
        self.subscription = self.create_subscription(
            Pose, "pose", self.pose_callback, 1
        )
        self.poses = []
        # Bit jank, but need to check if rosbag finished, use a timer to check this.
        # If flag is 0, then its been a second since last message, so kill self
        self.timer = self.create_timer(1, self.timer_callback)
        self.flag = 1

    def timer_callback(self):
        pu.db
        if self.flag:
            self.flag = 0
        else:
            self.destroy_node()
            return

    def pose_callback(self, msg):
        pu.db
        self.get_logger().info('Got pose for "%s"' % str(msg.header.stamp))
        self.poses.append( (msg.position.x, msg.position.y, msg.position.z, msg.orientation.w, msg.orietnation.x, msg.orientation.y, msg.orientation.z))
        self.flag = 1


def collect_data():
    rosbags = Path('./data/bags/').rglob('*.db3')
    for rosbag in rosbags:
        #In the future should check to make sure text file
        #doesn't already exist. Could probably also optimize this
        #by reseting orbslam node instead of making new instance
        parent = rosbag.parent
        pu.db
        pose_args = {"visualize": False}
        p2 = Process(
            target=run_node,
            args=(
                Orbslam2Pose,
                pose_args,
            ),
        )
        p2.start()
        p = Process(
            target=play_rosbag,
            args=(str(parent), False),
        )
        p.start()
        poseWriter = PoseWriter()
        rclpy.spin(poseWriter)
        pu.db
        os.kill(p.pid, signal.SIGKILL)
        os.kill(p2.pid, signal.SIGKILL)


if __name__ == "__main__":
    rclpy.init()
    collect_data()
    rclpy.shutdown()
