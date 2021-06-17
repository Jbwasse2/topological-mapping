# There seems to be a glitch with ROS2 Services where I can't create the second
# instance of a ROS Service, so it is created here first...
import rclpy


# Put stuff here for when testing starts
def pytest_sessionstart(session):
    rclpy.init()


# Put stuff here for when testing ends
def pytest_sessionfinish(session, exitstatus):
    rclpy.shutdown()
