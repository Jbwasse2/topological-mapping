# Methods for getting the pose of the robot
import pudb
import numpy as np
import quaternion

# I have no idea why, but in order for orbslam2 to run on my system
# I need to import cv2 first even if I don't use it.
import cv2  # noqa
import orbslam2
from rclpy.node import Node
import pudb  # noqa
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from std_msgs.msg import Header


# TODO implement this for RGB or RGBD
class Orbslam2Pose(Node):
    def __init__(
        self,
        vocab_path="./configs/ORBvoc.txt",
        slam_settings_path="./configs/mp3d3_small1k.yaml",
        visualize=False,
    ):
        super().__init__("orbslam2")
        self.slam = orbslam2.System(
            vocab_path, slam_settings_path, orbslam2.Sensor.MONOCULAR
        )
        self.slam.set_use_viewer(visualize)
        self.slam.initialize()
        self.slam.reset()
        self.start_time = self.get_clock().now().to_msg()
        self.subscription = self.create_subscription(
            Image, "camera", self.update_internal_state, 1
        )
        self.publisher = self.create_publisher(Pose, "pose", 10)
        self.bridge = CvBridge()
        self.get_logger().info("ORBSLAM2 succesfully started")

    # Takes in image as RGB and returns mono gray image to be added to orbslam2
    def mono_from_observation(self, observation):
        return np.dot(observation[..., :3], [0.2989, 0.5870, 0.1140])

    # Takes 13 pt trajectory_point from orbslam2 wrapper and converts them into pose
    # Format is as follows where Rwc is SE3 rotation and twc is position offset
    # Time
    # Rwc.at<float>(0,0),
    # Rwc.at<float>(0,1),
    # Rwc.at<float>(0,2),
    # twc.at<float>(0),
    # Rwc.at<float>(1,0),
    # Rwc.at<float>(1,1),
    # Rwc.at<float>(1,2),
    # twc.at<float>(1),
    # Rwc.at<float>(2,0),
    # Rwc.at<float>(2,1),
    # Rwc.at<float>(2,2),
    # twc.at<float>(2)
    def trajectory_point_to_pose(self, trajectory_point):
        trajectory_point = np.array(trajectory_point[1:]).reshape(3, 4)
        position = trajectory_point[:, 3]
        rot_se3 = trajectory_point[0:3, 0:3]
        rot_quat = quaternion.from_rotation_matrix(rot_se3)
        return position, rot_quat

    def update_internal_state(self, msg):
        assert msg.height == 480
        assert msg.width == 640
        assert msg.encoding == "bgr8"
        # Check if camera data is empty, if it is then skip this frame
        if sum(msg.data) == 0:
            self.get_logger().warning(
                'Received BLANK IMAGE, not updating ORBSLAM2 "%s"'
                % str(msg.header.stamp)
            )
            return
        image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        current_time_second = msg.header.stamp.sec - self.start_time.sec
        current_time_nanosecond = msg.header.stamp.nanosec - self.start_time.nanosec
        current_time = current_time_second + (current_time_nanosecond * 10 ** -9)
        self.slam.process_image_mono(image, current_time)
        if not str(self.slam.get_tracking_state()) == "OK":
            self.get_logger().warning(
                'ORBSLAM2 tracking state is not OK! "%s" "%s" "%s"'
                % (
                    str(self.slam.get_tracking_state()),
                    str(current_time),
                    str(msg.header.stamp),
                )
            )
        else:
            self.get_logger().info(
                'ORBSLAM2 succesfully updated state "%s"' % str(msg.header.stamp)
            )
            header = Header()
            header.frame_id = msg.header.frame_id
            msg = Pose()
            header.stamp = self.get_clock().now().to_msg()
            position, rotation = self.trajectory_point_to_pose(
                self.slam.get_trajectory_points()[-1]
            )
            msg.position.x = position[0]
            msg.position.y = position[1]
            msg.position.z = position[2]
            msg.orientation.x = rotation.x
            msg.orientation.y = rotation.y
            msg.orientation.z = rotation.z
            msg.orientation.w = rotation.w
            self.publisher.publish(msg)
