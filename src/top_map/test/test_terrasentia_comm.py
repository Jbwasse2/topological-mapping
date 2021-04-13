import rclpy
import os
import signal
from multiprocessing import Process
import time
from geometry_msgs.msg import Vector3
import socket
from rclpy.node import Node
from top_map.terrasentia_comm import TerraComm
from testing_helper import run_node
import pudb  # noqa

# This dummy will test the message passsing to the TerraSentia.
# This will open a socket on a localhost port,
# and test that message passing works over socket.


# Client must be opened before the server!
class TerraCommDummyClient(Node):
    def __init__(self):
        super().__init__("terra_dummy_twist_client")
        # Redirecting stdout to file in order to print if results are good
        with open("./test/results/client_results.out", "w") as f:
            f.write("Fail")
        serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        serversocket.bind(("localhost", 51717))
        serversocket.listen(1)  # become a server socket, maximum 1 connections
        self.get_logger().info("[TerraCommDummy] Create Dummy Client")
        connection, address = serversocket.accept()
        while True:
            buf = connection.recv(64)
            if len(buf) > 0:
                if "1.0,2.0,3.0,4.0,5.0,6.0" in str(buf):
                    with open("./test/results/client_results.out", "w") as f:
                        f.write("Success")
                break


class TerraCommDummyServer(TerraComm):
    def __init__(self):
        super().__init__(host="localhost")
        # Create server side socket
        linear = Vector3()
        angular = Vector3()

        linear.x = 1.0
        linear.y = 2.0
        linear.z = 3.0

        angular.x = 4.0
        angular.y = 5.0
        angular.z = 6.0

        terra_msg = self.create_terrasentia_message(linear, angular)
        self.get_logger().info("[TerraCommDummy] Create Dummy Server")
        NUMBER_OF_ATTEMPTS = 5
        for i in range(NUMBER_OF_ATTEMPTS):
            try:
                self.socket.sendall(bytes(terra_msg, "utf-8"))
            except BrokenPipeError as e:
                self.get_logger().info(str(e))
                break
            time.sleep(0.1)


# Test sending terrasentia messages over socket
# Results are contained in ./test/results/client_results.out
def test_terra_comm():
    # Start a dummy server to send messages to run in the background
    rclpy.init()

    client_args = {}
    p = Process(
        target=run_node,
        args=(
            TerraCommDummyClient,
            client_args,
        ),
    )
    p.start()
    # Need to give some time for process to open socket
    time.sleep(0.1)
    TerraCommDummyServer()
    os.kill(p.pid, signal.SIGKILL)
    rclpy.shutdown()
    with open("./test/results/client_results.out", "r") as f:
        results = f.readline()
    assert "Success" in results
