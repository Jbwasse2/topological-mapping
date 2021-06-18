import pytest
import signal
from multiprocessing import Process
import rclpy
from top_map.planner import Planner
import subprocess
from rclpy.task import Future
import os
from top_map.util import play_rosbag


class PlannerTester(Planner):
    def __init__(self, future, timeout=None):
        super().__init__("./data/indoorData/results/top_maps/test_top_map.pkl")
        self.future = future
        if timeout is not None:
            self.timer = self.create_timer(timeout, self.timer_callback)

    def timer_callback(self):
        if self.current_node != None and self.plan != None:
            self.future.set_result("Pass")
        else:
            self.future.set_result("Timeout")


def test_top_map():
    try:
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
        assert plannerTester.future.result() == "Pass"
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
