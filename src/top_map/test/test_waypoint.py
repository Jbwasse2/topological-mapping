from top_map.waypoint import WaypointPublisher
from multiprocessing import Process
from testing_helper import play_rosbag
import os
import signal
import rclpy
from rclpy.node import Node
import cv2

# Goal for meng is a list of 11 images where 6th image is goal and 5 images before and after give context
def get_goal():
    goal = []
    goal.append(cv2.imread("./test/resources/frame000365.png"))
    goal.append(cv2.imread("./test/resources/frame000366.png"))
    goal.append(cv2.imread("./test/resources/frame000367.png"))
    goal.append(cv2.imread("./test/resources/frame000368.png"))
    goal.append(cv2.imread("./test/resources/frame000369.png"))
    goal.append(cv2.imread("./test/resources/frame000370.png"))
    goal.append(cv2.imread("./test/resources/frame000371.png"))
    goal.append(cv2.imread("./test/resources/frame000372.png"))
    goal.append(cv2.imread("./test/resources/frame000373.png"))
    goal.append(cv2.imread("./test/resources/frame000374.png"))
    goal.append(cv2.imread("./test/resources/frame000375.png"))
    return goal


# TODO: I think I should go through the example made in rmp_nav to see if my method of finessing the data is correct.
def test_meng_wp_video():
    rclpy.init()
    # Run bag node to test
    rosbag_location = "./test/resources/rosbag/rosbag2_2021_04_14-09_01_00"
    p = Process(
        target=play_rosbag,
        args=(rosbag_location, False),
    )
    waypointPublisher = WaypointPublisher("./test/results/wp/")
    goal = get_goal()
    waypointPublisher.goal = goal
    waypointPublisher.goal_show = goal[6]
    p.start()
    try:
        rclpy.spin(waypointPublisher)
    except Exception as e:
        print(e)
    waypointPublisher.destroy_node()
    os.kill(p.pid, signal.SIGKILL)
    rclpy.shutdown()
    pu.db
