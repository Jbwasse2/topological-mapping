from rclpy.task import Future
import pytest
import subprocess
from top_map.util import play_rosbag
from top_map.topological_map import (
    TopologicalMap,
)
import os
import signal
from multiprocessing import Process
import rclpy
from copy import deepcopy


class TopMapTester(TopologicalMap):
    def __init__(self, future, timeout=None, use_pose_estimate=True):
        super().__init__(use_pose_estimate=use_pose_estimate)
        self.future = future
        if timeout is not None:
            self.timer = self.create_timer(timeout, self.timer_callback)

    def timer_callback(self):
        if len(self.map.nodes) > 0:
            self.future.set_result("Pass")
            self.save("./data/indoorData/results/top_maps/test_top_map.pkl")
        else:
            self.future.set_result("Timeout")


class LoopClosureTester(TopologicalMap):
    def __init__(self, future, timeout=None):
        super().__init__()
        self.future = future
        self.loop_closure_timer = self.create_timer(1, self.loop_closure_timer)
        if timeout is not None:
            self.timer = self.create_timer(timeout, self.timer_callback)

    def loop_closure_timer(self):
        old_map = deepcopy(self.map)
        self.loop_closure(self.map)

    def timer_callback(self):
        if len(self.map.nodes) > 0:
            self.future.set_result("Pass")
        else:
            self.future.set_result("Timeout")


@pytest.mark.skip(reason="writing other test")
def test_top_map():
    try:
        rosbag_location = "./test/testing_resources/rosbag/test_long.bag"
        p2 = Process(
            target=play_rosbag,
            args=(rosbag_location, False, ""),
        )
        p2.start()
        future = Future()
        topMapTester = TopMapTester(future, timeout=10, use_pose_estimate=True)
        rclpy.spin_until_future_complete(topMapTester, future)
    except Exception as e:
        raise (e)
    else:
        topMapTester.destroy_node()
        assert topMapTester.future.result() == "Pass"
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


def test_loop_closure():
    try:
        future = Future()
        topMapTester = LoopClosureTester(future, timeout=3)
        topMapTester.load(
            "./data/indoorData/results/top_maps/test_loop_closure_map.pkl"
        )
        rclpy.spin_until_future_complete(topMapTester, future)
    except Exception as e:
        raise (e)
    else:
        topMapTester.destroy_node()
        assert topMapTester.future.result() == "Pass"
    finally:
        pass
