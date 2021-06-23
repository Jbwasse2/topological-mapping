import pytest
import signal
from multiprocessing import Process
import rclpy
from top_map.planner import Planner
from top_map.waypoint import WaypointPublisher
import subprocess
from rclpy.task import Future
import os
from top_map.util import play_rosbag, run_node
from geometry_msgs.msg import TwistStamped


class PlannerTester(Planner):
    def __init__(self, future, timeout=None):
        super().__init__("./data/indoorData/results/top_maps/test_top_map.pkl")
        self.future = future
        self.sub_ = self.create_subscription(
            TwistStamped, "/terrasentia/cmd_vel", self.twistcallback
        )
        self.twist_called = False
        if timeout is not None:
            self.timer = self.create_timer(timeout, self.timer_callback)

    def twistcallback(self, msg):
        # I think that some messages are sent from some other part
        # of the ros websocket. But hey are always 0's
        if (
            msg.twist.linear.x == 0.0
            and msg.twist.linear.y == 0.0
            and msg.twist.linear.z == 0.0
            and msg.twist.angular.x == 0.0
            and msg.twist.angular.y == 0.0
            and msg.twist.angular.z == 0.0
        ):
            pass
        else:
            self.twist_called = True

    def timer_callback(self):
        if self.current_node is not None and self.plan is not None:
            self.future.set_result("Pass")
        else:
            self.future.set_result("Timeout")


def test_top_map():
    try:
        # Start Waypoint Publisher
        args = {"create_graphic": "./test/results/planner/"}
        p = Process(
            target=run_node,
            args=(
                WaypointPublisher,
                args,
            ),
        )
        p.start()
        # Start ros bag
        rosbag_location = "./test/testing_resources/rosbag/test_long.bag"
        p2 = Process(
            target=play_rosbag,
            args=(rosbag_location, False, ""),
        )
        p2.start()
        future = Future()
        plannerTester = PlannerTester(future, timeout=5)
        rclpy.spin_until_future_complete(plannerTester, future)
    except Exception as e:
        raise (e)
    else:
        plannerTester.destroy_node()
        if plannerTester.future.result() != "Pass":
            pytest.fail("Future is " + str(plannerTester.future.result()))
        if not plannerTester.twist_called:
            pytest.fail("Twist was never called! Did you set a goal?")
    finally:
        # kills test.bag
        kill_testbag_cmd = (
            "export PYTHONPATH= && . /opt/ros/melodic/setup.sh && rosnode list "
            + "| grep play | xargs rosnode kill"
        )
        subprocess.Popen(
            kill_testbag_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            shell=True,
        )
        os.kill(p2.pid, signal.SIGKILL)
