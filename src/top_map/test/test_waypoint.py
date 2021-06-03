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
    goal.append(cv2.imread("./test/testing_resources/000274.png"))
    goal.append(cv2.imread("./test/testing_resources/000275.png"))
    goal.append(cv2.imread("./test/testing_resources/000276.png"))
    goal.append(cv2.imread("./test/testing_resources/000277.png"))
    goal.append(cv2.imread("./test/testing_resources/000278.png"))
    goal.append(cv2.imread("./test/testing_resources/000279.png"))
    goal.append(cv2.imread("./test/testing_resources/000280.png"))
    goal.append(cv2.imread("./test/testing_resources/000281.png"))
    goal.append(cv2.imread("./test/testing_resources/000282.png"))
    goal.append(cv2.imread("./test/testing_resources/000283.png"))
    goal.append(cv2.imread("./test/testing_resources/000284.png"))
    return goal


# TODO: I think I should go through the example made in rmp_nav to see
# if my method of finessing the data is correct.
def test_meng_wp_video():
    rclpy.init()
    # Run bag node to test
    rosbag_location = "./test/testing_resources/rosbag/test.bag"
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
