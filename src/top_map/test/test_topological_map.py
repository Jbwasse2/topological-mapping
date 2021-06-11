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
from mock import patch


class TopMapTester(TopologicalMap):
    def __init__(self, future):
        super().__init__()
        self.future = future


def test_get_embedding():
    embeddingGetter = EmbeddingGetter()
    random_image = np.random.randint(255, size=(1, 3, 224, 224))
    random_image = torch.from_numpy(random_image).float()
    embedding = embeddingGetter(random_image)
    assert embedding.shape == torch.Size([1, 512, 7, 7])


def test_get_classification():
    embeddingClassifier = EmbeddingsClassifier()
    embedding1 = np.random.rand(1, 512, 7, 7)
    embedding2 = np.random.rand(1, 512, 7, 7)
    embedding1 = torch.from_numpy(embedding1).float()
    embedding2 = torch.from_numpy(embedding2).float()
    prediction = embeddingClassifier(embedding1, embedding2)
    assert prediction.shape == torch.Size([1, 2])


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
        topMapTester = TopMapTester(future)
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
        # assert results == 1
