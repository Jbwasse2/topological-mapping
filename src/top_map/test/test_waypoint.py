import glob
import os
import signal
import subprocess
from multiprocessing import Process

import cv2
import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.task import Future
from top_map.util import play_rosbag
from top_map.waypoint import WaypointPublisher


class WaypointPublisherTester(WaypointPublisher):
    def __init__(self, timeout, future):
        super().__init__()
        self.future = future
        self.timer = self.create_timer(timeout, self.timer_callback)
        self.sub_ = self.create_subscription(TwistStamped,
                                             "/terrasentia/cmd_vel",
                                             self.twistcallback, 2)

    def twistcallback(self, msg):
        x = msg.twist.linear.x
        y = msg.twist.linear.y
        z = msg.twist.linear.z
        a = msg.twist.angular.x
        b = msg.twist.angular.y
        c = msg.twist.angular.z
        if x != 0.0 or y != 0.0 or z != 0.0 or a != 0.0 or b != 0.0 or c != 0.0:
            print(x, y, z, a, b, c)
            self.twistflag = True

    def timer_callback(self):
        self.future.set_result("Timeout")


# Goal for meng is a list of 11 images where 6th image is goal and 5
# images before and after give context
def get_goal():
    goal = []
    goal.append(cv2.imread("./test/testing_resources/0000274.jpg"))
    goal.append(cv2.imread("./test/testing_resources/0000275.jpg"))
    goal.append(cv2.imread("./test/testing_resources/0000276.jpg"))
    goal.append(cv2.imread("./test/testing_resources/0000277.jpg"))
    goal.append(cv2.imread("./test/testing_resources/0000278.jpg"))
    goal.append(cv2.imread("./test/testing_resources/0000279.jpg"))
    goal.append(cv2.imread("./test/testing_resources/0000280.jpg"))
    goal.append(cv2.imread("./test/testing_resources/0000281.jpg"))
    goal.append(cv2.imread("./test/testing_resources/0000282.jpg"))
    goal.append(cv2.imread("./test/testing_resources/0000283.jpg"))
    goal.append(cv2.imread("./test/testing_resources/0000284.jpg"))
    return goal


# TODO: I think I should go through the example made in rmp_nav to see
# if my method of finessing the data is correct.
def test_meng_wp_video():
    # Run bag node to test
    rosbag_location = "./test/testing_resources/rosbag/test.bag"
    p = Process(
        target=play_rosbag,
        args=(
            rosbag_location,
            False,
            "--topics /terrasentia/usb_cam_node/image_raw --wait-for-subscribers",
        ),
    )
    future = Future()
    waypointPublisher = WaypointPublisherTester(5, future)
    goal = get_goal()
    waypointPublisher.goal = goal
    waypointPublisher.goal_show = goal[6]
    p.start()
    rclpy.spin_until_future_complete(waypointPublisher, future)
    waypointPublisher.destroy_node()
    kill_testbag_cmd = (
        "export PYTHONPATH= && . /opt/ros/melodic/setup.sh && rosnode list " +
        "| grep play | xargs rosnode kill")
    subprocess.Popen(
        kill_testbag_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        shell=True,
    )
    os.kill(p.pid, signal.SIGKILL)

    assert len(glob.glob("./test/results/wp/*")) != 0
    assert waypointPublisher.twistflag
