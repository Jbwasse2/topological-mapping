from top_map.similarityService import SimilarityService
from rclpy.task import Future
import subprocess
from top_map.util import play_rosbag
from top_map.topological_map import (
    TopologicalMap,
)
import os
import signal
from multiprocessing import Process
from top_map.util import run_node
import rclpy


class TopMapTester(TopologicalMap):
    def __init__(self, future, timeout=None):
        super().__init__()
        self.future = future
        if timeout is not None:
            self.timer = self.create_timer(timeout, self.timer_callback)

    def timer_callback(self):
        pu.db
        if len(self.map.nodes) > 0:
            self.future.set_result("Pass")
            self.save("./data/indoorData/results/top_maps/test_top_map.pkl")
        else:
            self.future.set_result("Timeout")


def test_top_map():
    try:
        rclpy.init()
        args = {}
        p = Process(
            target=run_node,
            args=(SimilarityService, args),
        )
        p.start()
        rosbag_location = "./test/testing_resources/rosbag/test_long.bag"
        p2 = Process(
            target=play_rosbag,
            args=(rosbag_location, False, ""),
        )
        p2.start()
        future = Future()
        topMapTester = TopMapTester(future, timeout=70)
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
        # Kill similarity publisher, for some reason it is sticking around
        os.kill(p.pid, signal.SIGKILL)
        os.kill(p2.pid, signal.SIGKILL)

        rclpy.shutdown()
