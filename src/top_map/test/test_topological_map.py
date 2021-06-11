from top_map.similarityService import SimilarityService
from rclpy.task import Future
import torch
import subprocess
from top_map.util import play_rosbag
from top_map.topological_map import (
    TopologicalMap,
    EmbeddingGetter,
    EmbeddingsClassifier,
)
import os
import signal
from multiprocessing import Process
from top_map.util import run_node
from top_map_msg_srv.srv import Similarity
import rclpy
import numpy as np
from cv_bridge import CvBridge
from rclpy.node import Node


class TopMapTester(TopologicalMap):
    def __init__(self, future, timeout=None):
        super().__init__()
        self.future = future
        if timeout is not None:
            self.timer = self.create_timer(timeout, self.timer_callback)
    def timer_callback(self):
        if len(self.map.nodes) > 0:
            self.future.set_result("Pass")
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
            args=(rosbag_location, False, "-l"),
        )
        p2.start()
        future = Future()
        topMapTester = TopMapTester(future, timeout=2)
        rclpy.spin_until_future_complete(topMapTester, future)
    except Exception as e:
        results = 0
        print(e)
    else:
        pass
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
        os.kill(p.pid, signal.SIGKILL)
        os.kill(p2.pid, signal.SIGKILL)
        topMapTester.destroy_node()
        rclpy.shutdown()
        assert topMapTester.future.result() == "Pass"
