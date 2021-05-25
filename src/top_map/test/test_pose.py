import os
import signal
from multiprocessing import Process

import rclpy
from geometry_msgs.msg import Pose
from rclpy.node import Node
from testing_helper import play_rosbag, run_node
from top_map.pose import Orbslam2Pose


class PoseTester(Node):
    def __init__(self):
        super().__init__("pose_tester")
        self.subscription = self.create_subscription(
            Pose, "pose", self.pose_callback, 1
        )
        self.results = False
        self.timer = self.create_timer(30, self.timer_callback)

    def pose_callback(self, msg):
        self.results = True

    def timer_callback(self):
        self.destroy_node()


# Running this test seems to break the other one.
# TODO figure out why
# def test_tp_to_pose():
#    rclpy.init()
#    tp = (100, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
#    pose = Orbslam2Pose(visualize=False)
#    position, rotation = pose.trajectory_point_to_pose(tp)
#    assert position[0] == 0.0 and position[1] == 0.0 and position[2] == 0.0
#    assert (
#        rotation.w == 1.0
#        and rotation.x == 0.0
#        and rotation.y == 0.0
#        and rotation.z == 0.0
#    )
#    rclpy.shutdown()


# Check to make sure a pose message gets published
def test_orbslam2():
    rclpy.init()
    rosbag_location = "./test/testing_resources/rosbag/rosbag2_2021_04_14-09_01_00"
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
        args=(rosbag_location, False),
    )
    p.start()
    pose = PoseTester()
    rclpy.spin_once(pose)
    try:
        pose.destroy_node()
    except Exception as e:
        print(e)
    os.kill(p.pid, signal.SIGKILL)
    os.kill(p2.pid, signal.SIGKILL)
    rclpy.shutdown()
    assert pose.results is True
