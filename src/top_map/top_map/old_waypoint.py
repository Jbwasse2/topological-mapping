import os
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pudb
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point32, Twist, Vector3
from rclpy.node import Node
from rmp_nav.common.utils import (get_gibson_asset_dir, get_project_root,
                                  pprint_dict, str_to_dict)
from rmp_nav.simulation import agent_factory, sim_renderer
from sensor_msgs.msg import Image
from topological_nav.reachability import model_factory

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
matplotlib.use('TkAgg') 

class WaypointPublisher(Node):

    def __init__(self, create_graphic=False):
        super().__init__('waypoint_publisher')
        #If output file doesn't exist create it for iamges to be saved to instead of displayed.
        if create_graphic:
            Path("./output").mkdir(parents=True, exist_ok=True)
            self.counter = 0

        self.create_graphic = create_graphic
        self.model = self.get_model()
        self.subscription = self.create_subscription(Image, 'camera', self.image_callback, 1)
        self.publisher_ = self.create_publisher(Twist, 'terra_command_twist', 1)
        self.subscription = self.create_subscription(Image, 'goal', self.goal_callback, 1)
        self.bridge = CvBridge()
        self.get_logger().info('Created Waypoint Node')
        self.goal = None
        self.goal_show = None
        self.path_index = 0
        self.trajectories = []

    def get_model(self):
        model = model_factory.get("model_12env_v2_future_pair_proximity_z0228")(
            device="cuda"
        )
        return model

    def goal_callback(self, msg):
        #This callback takes a collection of images sent from the GoalPublisher that it beleives is the trajectory the robot should follow
        #The received image should be stacked images ie if there are 20 64x64 images, then the resulting image should be 1280x64.
        self.get_logger().info('[waypoint.py] Trajectory Images Found!')
        image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        image = image / 255
        IMAGE_HEIGHT = 64
        number_of_images = int(msg.height / IMAGE_HEIGHT)
        for i in range(number_of_images):
            self.trajectories.append(image[IMAGE_HEIGHT*i:IMAGE_HEIGHT*(i+1), :])
        self.path_index = 5
        self.update_goal()

    def update_goal(self):
        #Get 11 images centered around path_index
        pu.db
        self.goal = []
        for i in range(self.path_index-5, self.path_index+6):
            self.goal.append(self.trajectories[i])
        self.goal_show = (255*self.goal[5]).astype(np.uint8)
        self.goal_show = cv2.cvtColor(self.goal_show, cv2.COLOR_RGB2BGR)


    #No longer used, goal images are sent from the GoalPublisher
    def set_goal(self):
        def get_img(name):
            a = cv2.resize(
                cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB),
                dsize=(64, 64),
                interpolation=cv2.INTER_CUBIC)
            return a

        self.goal = []

        self.goal.append(get_img('./data/20200914_083340.jpg'))
        self.goal.append(get_img('./data/20200914_083342.jpg'))
        self.goal.append(get_img('./data/20200914_083344.jpg'))
        self.goal.append(get_img('./data/20200914_083346.jpg'))
        self.goal.append(get_img('./data/20200914_083347.jpg'))
        self.goal.append(get_img('./data/20200914_083348.jpg'))
        self.goal.append(get_img('./data/20200914_083350.jpg'))
        self.goal.append(get_img('./data/20200914_083351.jpg'))
        self.goal.append(get_img('./data/20200914_083353.jpg'))
        self.goal.append(get_img('./data/20200914_083354.jpg'))
        self.goal.append(get_img('./data/20200914_083356.jpg'))
        self.goal_show = cv2.cvtColor(self.goal[6], cv2.COLOR_RGB2BGR)

    def create_waypoint_message(self, waypoint, reachability_estimator):
        lin = Vector3()
        angular = Vector3()
        #item converts from numpy float type to python float type
        lin.x = float(waypoint[0].item()) / 4
        lin.y = 0.0
        #msg.z = reachability_estimator.item()
        lin.z = 0.0
        angular.x = 0.0
        angular.y = 0.0
        angular.z = float(waypoint[1].item()) / 4
        msg = Twist()
        msg.linear = lin
        msg.angular = angular
        return msg

    def image_callback(self, msg):
        self.get_logger().info('I heard {0}'.format(str(msg.header)))
        image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #The above converts the image to RGB
        if self.goal is None:
            self.get_logger().info("[waypoint.py] No goal image, waiting for planner to finish")
            return
        pu.db
        waypoint, reachability_estimator = self.get_wp(image/255, self.goal)
        self.get_logger().info("Reachability Estimator is {0}".format(str(reachability_estimator)))
        if reachability_estimator >= 0.95:
            self.path_index += 1
            self.update_goal()
        msg = self.create_waypoint_message(waypoint, reachability_estimator)
        self.publisher_.publish(msg)
        #The arrow is strictly for visualization, not used for navigation
        if self.create_graphic:
            arrow_start = (32,20)
            arrow_end = (32 + int(10 * -waypoint[1]), 20 + int(10 * -waypoint[0]))
            color = (0, 0, 255) #Red
            thickness = 2
            #cv2 like BGR because they like eating glue
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = np.hstack((image, self.goal_show))
            image = cv2.arrowedLine(image, arrow_start, arrow_end, color, thickness)
            (height, width, _) = image.shape
#            image = cv2.resize(image, dsize=(width * 10, height * 10), interpolation=cv2.INTER_CUBIC)
            #Add 0's to front to make it easier for script to make into video, counter should not be larger than 6 digits (IE 999999)
            counter_string = str(self.counter).rjust(6, '0')
            cv2.imwrite('./output/frame' + counter_string + '.png', image)
            self.counter += 1


    def get_wp(self, ob, goal):
        agent = agent_factory.agents_dict[self.model["agent"]]()
        follower = self.model["follower"]
        goal = self.cv2_to_model_im(goal)
        ob = self.cv2_to_model_im(ob)
        return (
            follower.motion_policy.predict_waypoint(ob, goal),
            follower.sparsifier.predict_reachability(ob, goal),
        )

    def show_img(self, img):
        matplotlib.use('TkAgg') 
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 0)
        plt.imshow(img)
        plt.show()

    #Cv2 gives images in BGR, and from 0-255
    #We want RGB and from 0-1
    #Can also get list/ np array of images, this should be handled
    def cv2_to_model_im(self,im):
        im = np.asarray(im)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        assert len(im.shape) == 3 or len(im.shape) == 4
        if len(im.shape) == 3:
            im = np.swapaxes(im, 0, 2)
            im = np.swapaxes(im, 1, 2)
            im = np.asarray(im)
#            self.show_img(im)
            im = (im / 255).astype("float32")
        else:
            im = np.swapaxes(im, 1, 3)
            im = np.swapaxes(im, 2, 3)
            im = np.asarray(im)
#            self.show_img(im[0])
            im = (im / 255).astype("float32")
        return im


    def __del__(self):
        cv2.destroyAllWindows()
        self.destroy_node()
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)

    waypoint_publisher = WaypointPublisher(create_graphic=True)

    rclpy.spin(waypoint_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    waypoint_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
