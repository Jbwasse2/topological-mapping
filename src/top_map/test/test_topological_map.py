import os
import signal
import subprocess
from multiprocessing import Process

import numpy as np
import rclpy
from rclpy.task import Future

from top_map.topological_map import TopologicalMap
from top_map.util import play_rosbag


class TopMapTester(TopologicalMap):
    def __init__(self, future, timeout=None):
        super().__init__()
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
        self.loop_closure(self.map)

    def timer_callback(self):
        if len(self.map.nodes) > 0:
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
        topMapTester = TopMapTester(future, timeout=10)
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
    class position:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z

    try:
        future = Future()
        topMapTester = LoopClosureTester(future, timeout=3)
        # Set ekf_pose position
        topMapTester.ekf_pose[(0, 0)] = {}
        topMapTester.ekf_pose[(0.0, 0)]["position"] = position(0.0, 0.0, 0.0)
        topMapTester.ekf_pose[(0.0, 1)] = {}
        topMapTester.ekf_pose[(0.0, 1)]["position"] = position(0.0, 2.0, 0.0)
        topMapTester.ekf_pose[(0.0, 2)] = {}
        topMapTester.ekf_pose[(0.0, 2)]["position"] = position(0.0, 0.1, 0.0)
        # Set nodes in test map
        topMapTester.map.add_node((0, 0))
        topMapTester.map.add_node((0, 1))
        topMapTester.map.add_node((0, 2))
        # Set edges in test map
        topMapTester.map.add_edge((0, 1), (0, 0))
        topMapTester.map.add_edge((0, 2), (0, 0))
        # Set embeddings
        embedding = np.random.rand(1, 512, 7, 7).astype(np.float32)
        topMapTester.embedding_dict[(0, 0)] = embedding
        topMapTester.embedding_dict[(0, 1)] = embedding
        topMapTester.embedding_dict[(0, 2)] = embedding
        rclpy.spin_until_future_complete(topMapTester, future)
    except Exception as e:
        raise (e)
    else:
        topMapTester.destroy_node()
        nodes = topMapTester.map.nodes
        assert len(nodes) == 2
        assert (0, 0) in nodes
        assert (0, 1) in nodes
        edges = topMapTester.map.edges
        assert ((0, 1), (0, 0)) in edges
    finally:
        pass
