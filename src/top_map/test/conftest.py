# There seems to be a glitch with ROS2 Services where I can't create the second
# instance of a ROS Service, so it is created here first...
import signal
import os
import rclpy
from multiprocessing import Process
from top_map.util import run_node
from top_map.similarityService import SimilarityService

p = None


def pytest_sessionstart(session):
    rclpy.init()
    args = {}
    global p
    p = Process(
        target=run_node,
        args=(SimilarityService, args),
    )
    p.start()


def pytest_sessionfinish(session, exitstatus):
    os.kill(p.pid, signal.SIGKILL)
    rclpy.shutdown()
