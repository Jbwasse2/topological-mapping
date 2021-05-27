import glob
import os
import signal
from multiprocessing import Process

import cv2

import rclpy
from rclpy.task import Future
from top_map.util import play_rosbag
from top_map.waypoint import WaypointPublisher


class WaypointPublisherTester(WaypointPublisher):
    def __init__(self, timeout, create_graphic, future):
        super().__init__(create_graphic)
        self.future = future
        self.timer = self.create_timer(timeout, self.timer_callback)

    def timer_callback(self):
        self.future.set_result("Timeout")


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
def test_meng_wp_video():
    rclpy.init()
    # Run bag node to test
    rosbag_location = "./test/testing_resources/rosbag/rosbag2_2021_04_14-09_01_00"
    p = Process(
        target=play_rosbag,
        args=(rosbag_location, False),
    )
    future = Future()
    waypointPublisher = WaypointPublisherTester(3, "./test/results/wp/", future)
    goal = get_goal()
    waypointPublisher.goal = goal
    waypointPublisher.goal_show = goal[6]
    p.start()
    rclpy.spin_until_future_complete(waypointPublisher, future)
    waypointPublisher.destroy_node()
    os.kill(p.pid, signal.SIGKILL)
    rclpy.shutdown()

    assert len(glob.glob("./test/results/wp/*")) != 0
