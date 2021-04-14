from top_map.pose import Orbslam2Pose
import rclpy
import os
import signal
from multiprocessing import Process
from testing_helper import play_rosbag


def test_orbslam2():
    rclpy.init()
    rosbag_location = "./test/rosbag/rosbag2_2021_04_14-09_01_00"
    p = Process(
        target=play_rosbag,
        args=(rosbag_location, False),
    )
    pose = Orbslam2Pose(visualize=False)
    p.start()
    rclpy.spin(pose)
    os.kill(p.pid, signal.SIGKILL)
    rclpy.shutdown()
    assert pose.results == True
