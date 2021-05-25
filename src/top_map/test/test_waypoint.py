import glob
import os
import signal
from multiprocessing import Process

import cv2
import pytest

import rclpy
from testing_helper import play_rosbag
from top_map.waypoint import WaypointPublisher


class WaypointPublisherTester(WaypointPublisher):
    def __init__(self, timeout, create_graphic):
        super().__init__(create_graphic)
        if timeout is not None:
            self.timer = self.create_timer(timeout, self.timer_callback)

    def timer_callback(self):
        self.destroy_node()


# Goal for meng is a list of 11 images where 6th image is goal and 5
# images before and after give context
def get_goal():
    goal = []
    goal.append(cv2.imread("./test/testing_resources/frame000365.png"))
    goal.append(cv2.imread("./test/testing_resources/frame000366.png"))
    goal.append(cv2.imread("./test/testing_resources/frame000367.png"))
    goal.append(cv2.imread("./test/testing_resources/frame000368.png"))
    goal.append(cv2.imread("./test/testing_resources/frame000369.png"))
    goal.append(cv2.imread("./test/testing_resources/frame000370.png"))
    goal.append(cv2.imread("./test/testing_resources/frame000371.png"))
    goal.append(cv2.imread("./test/testing_resources/frame000372.png"))
    goal.append(cv2.imread("./test/testing_resources/frame000373.png"))
    goal.append(cv2.imread("./test/testing_resources/frame000374.png"))
    goal.append(cv2.imread("./test/testing_resources/frame000375.png"))
    return goal


# TODO: I think I should go through the example made in rmp_nav to see
# if my method of finessing the data is correct.
@pytest.mark.skip(reason="Test hangs instead of fails if current time is used")
def test_meng_wp_video():
    rclpy.init()
    # Run bag node to test
    rosbag_location = "./test/testing_resources/rosbag/rosbag2_2021_04_14-09_01_00"
    p = Process(
        target=play_rosbag,
        args=(rosbag_location, False),
    )
    waypointPublisher = WaypointPublisherTester(3, "./test/results/wp/")
    goal = get_goal()
    waypointPublisher.goal = goal
    waypointPublisher.goal_show = goal[6]
    p.start()
    rclpy.spin(waypointPublisher)
    # TODO For some reason the code is not returning here!
    try:
        waypointPublisher.destroy_node()
    # Already killed
    except rclpy.handle.InvalidHandle:
        pass
    os.kill(p.pid, signal.SIGKILL)
    rclpy.shutdown()

    assert len(glob.glob("./test/results/wp/*")) == 1240
