import glob
import gzip
import math
import random
import time
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pudb
import quaternion
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as transforms
from torchvision import utils
from torchvision.transforms import ToTensor
from tqdm import tqdm

from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import (angle_between_quaternions,
                                          quaternion_from_coeff,
                                          quaternion_rotate_vector)
from rich.progress import track


def get_dict(fname):
    f = gzip.open(
        "../../data/datasets/pointnav/gibson/v4/train_large/content/"
        + fname
        + ".json.gz"
    )
    content = f.read()
    content = content.decode()
    content = content.replace("null", "None")
    content = eval(content)
    return content["episodes"]


class GibsonDataset(Dataset):
    """ Dataset collect from Habitat """

    def __init__(
        self,
        split_type,
        seed,
        angle_boxes=12,
        distance_boxes=15,
        visualize=False,
        episodes=100,
        max_distance=1,
        samples=10000,
        ignore_0=False,
        debug=False,
    ):
        random.seed(seed)
        self.angle_boxes = angle_boxes
        self.distance_boxes = distance_boxes
        self.debug = debug
        self.ignore_0 = ignore_0
        self.split_type = split_type
        self.max_distance = max_distance
        self.samples = samples
        # This is changed in the train_large_data.py file
        self.episodes = episodes
        self.image_data_path = (
            "../../data/datasets/pointnav/gibson/v4/train_large/images/"
        )
        self.dict_path = "../../data/datasets/pointnav/gibson/v4/train_large/content/"
        self.env_names = self.get_env_names()
        random.shuffle(self.env_names)
        split = 0.85
        self.train_env = self.env_names[0 : int(len(self.env_names) * split)]
        self.test_env = self.env_names[int(len(self.env_names) * split) :]
        if split_type == "train":
            self.labels = self.get_labels_dicts(self.train_env)
            self.dataset = self.get_dataset(self.train_env)
        elif split_type == "test":
            self.labels = self.get_labels_dicts(self.test_env)
            self.dataset = self.get_dataset(self.test_env)
        self.dataset = self.balance_dataset(samples)
        self.dataset = self.flatten_dataset()

    #        self.verify_dataset()

    def get_labels_dicts(self, envs):
        if self.debug:
            envs = envs[0:3]
        ret = {}
        self.distances = []
        self.angles = []
        for env in tqdm(envs):
            ret[env] = get_dict(env)
            # Get distance/rotation, Following is for debugging.
            rot_flag, disp_flag = 1, 1
            for i in range(len(ret[env][0]["shortest_paths"][0][0]) - 1):
                traj = ret[env][0]["shortest_paths"][0][0][i]
                next_traj = ret[env][0]["shortest_paths"][0][0][i + 1]
                if disp_flag:
                    if traj["action"] == 1:
                        dist = np.linalg.norm(
                            np.array(traj["position"]) - np.array(next_traj["position"])
                        )
                        self.distances.append(dist)
                        disp_flag = 0
                if rot_flag:
                    if traj["action"] == 2 or traj["action"] == 3:
                        q1 = quaternion.quaternion(*traj["rotation"])
                        q2 = quaternion.quaternion(*next_traj["rotation"])
                        angle = angle_between_quaternions(q1, q2)
                        self.angles.append(angle)
                        rot_flag = 0
        return ret

    def save_env_data(self, path):
        np.save(path + "train_env.npy", self.train_env)
        np.save(path + "test_env.npy", self.test_env)

    # Make sure all of the images exist
    def verify_dataset(self):
        for data in self.dataset:
            env, episode, l1, l2 = data
            img_file1 = Path(
                self.image_data_path
                + env
                + "/"
                + "episodeRGB"
                + str(episode)
                + "_"
                + str(l1).zfill(5)
                + ".jpg"
            )
            img_file2 = Path(
                self.image_data_path
                + env
                + "/"
                + "episodeRGB"
                + str(episode)
                + "_"
                + str(l2).zfill(5)
                + ".jpg"
            )
            assert img_file1.is_file()
            assert img_file2.is_file()
        # Make sure all keys have same amount of data and are in the correct range
        ret = {}
        for data in self.dataset:
            key = np.abs(data[3] - data[2])
            if key >= self.max_distance:
                key = self.max_distance
            if key <= -self.max_distance:
                key = -self.max_distance
            if key not in ret:
                ret[key] = 0
            ret[key] += 1
        number_of_1s = ret[list(ret.keys())[1]]
        range_of_keys = set(range(self.max_distance + 1))
        if self.ignore_0:
            range_of_keys.remove(0)
        for key, value in ret.items():
            assert key in range(self.max_distance + 1)
            assert value == number_of_1s
            range_of_keys.remove(key)
        assert range_of_keys == set()

    def flatten_dataset(self):
        # Dataset comes in as dict, would like it as a huge list of tuples
        ret = []
        for i in range(self.angle_boxes):
            for key, value in self.dataset[i].items():
                ret = ret + value
        random.shuffle(ret)
        return ret

    def balance_dataset(self, max_number_of_examples):
        min_size = np.inf
        for i in range(self.angle_boxes):
            for j in range(self.distance_boxes):
                min_size = min(min_size, len(self.dataset[i][j]))
        min_size = min(min_size, max_number_of_examples)
        ret = {}
        print("MIN SIZE = " + str(min_size))
        for i in range(self.angle_boxes):
            ret[i] = {}
            for j in range(self.distance_boxes):
                ret[i][j] = random.sample(self.dataset[i][j], min_size)
        return ret

    # Don't forget map/trajectory is directed.
    def get_dataset(self, envs):
        ret = {}
        for i in range(self.angle_boxes):
            ret[i] = {}
            for j in range(self.distance_boxes):
                ret[i][j] = []
        if self.debug:
            envs = envs[0:3]
        for env in tqdm(envs):
            d = self.labels[env]
            episodes = range(len(self.labels[env]))
            episode_len = {}
            for episode in episodes:
                episode_len[episode] = len(
                    self.labels[env][episode]["shortest_paths"][0][0]
                )
            sample_counter = 0
            while sample_counter < self.samples:
                episode_1 = random.choice(episodes)
                sample_1 = (episode_1, random.choice(range(episode_len[episode_1])))
                episodes_possible = list(episodes)
                episodes_possible.remove(episode_1)
                episode_2 = random.choice(episodes_possible)
                sample_2 = (episode_2, random.choice(range(episode_len[episode_2])))
                # Norm distance because distance only works on x-z plane I think>
                norm_distance, distance, angle = get_displacement_label(
                    sample_1, sample_2, d
                )

                if (
                    distance < self.max_distance
                    and np.abs(self.get_heading_diff(sample_1, sample_2, env)) < 0.55
                    and norm_distance < self.max_distance + 0.01
                ):
                    i, j = self.dist_angle_to_indicies(distance, angle)
                    if self.debug:
                        self.over_head_visualization(sample_1, sample_2, d, angle, env)
#                    if sample_1 == (16,339) and sample_2 == (11,359):
#                        self.over_head_visualization(sample_1, sample_2, d, angle, env)
                    ret[i][j].append((env, sample_1, sample_2, distance, angle))
                    sample_counter += 1
        return ret

    def get_cart_goal(self, local_start, local_goal, d):
        # See https://github.com/facebookresearch/habitat-lab/blob/b7a93bc493f7fb89e5bf30b40be204ff7b5570d7/habitat/tasks/nav/nav.py
        # for more information
        pu.db
        pos_start, rot_start = get_node_pose(local_start, d)
        pos_goal, rot_goal = get_node_pose(local_goal, d)
        # Quaternion is returned as list, need to change datatype
        direction_vector = pos_goal - pos_start
        norm_distance = np.linalg.norm(direction_vector)
        return np.array([norm_distance, -direction_vector[2], direction_vector[0]])

    def over_head_visualization(self, sample_1, sample_2, d, angle, env):
        print("**********************************************")
        print(env)
        print(sample_1, sample_2)
        pos_start = np.array([2, 0, -1])
        rot_start = [0.0, 0.924, 0.0, 0.383]
        #        rot_start = [0.0, -0.83, 0, 0.56]
        pos_goal = np.array([1, 0.0, -1])
        rot_goal = [0.0, 0.7070, 0.0, 0.7070]
        d[-1]["shortest_paths"][0][0][-1] = {
            "position": pos_start,
            "rotation": rot_start,
        }
        d[-2]["shortest_paths"][0][0][-2] = {"position": pos_goal, "rotation": rot_goal}
        #sample_1 = (-1, -1)
        #sample_2 = (-2, -2)
        # sample_1 = (11, 408)
        # sample_2 = (3, 281)
        pu.db
        norm_distance, distance, angle = get_displacement_label(sample_1, sample_2, d)
        pos_1, rot_1 = get_node_pose(sample_1, d)
        pos_2, rot_2 = get_node_pose(sample_2, d)
        pitch_1 = quaternion_to_yaw(rot_1)
        pitch_2 = quaternion_to_yaw(rot_2)
        # Translate so pos_1 is at center
        pos_2 = pos_2 - pos_1
        pos_1 = pos_1 - pos_1
        print(pos_1, pos_2)
        print(rot_1, rot_2)
        plt.plot([-pos_1[2], -pos_2[2]], [pos_1[0], pos_2[0]], "o", color="black")
        plt.xlim(-1.6, 1.6)
        plt.ylim(-1.6, 1.6)
        # plt.title(str(x) + " " + str(y))
        plt.title("Angle = " + str(angle) + " Dist = " + str(distance))
        # Get arrows of heading
        arrow_1_dx = 1.5 * np.cos(pitch_1)
        arrow_1_dy = 1.5 * np.sin(pitch_1)
        arrow_2_dx = 0.2 * np.cos(pitch_2)
        arrow_2_dy = 0.2 * np.sin(pitch_2)
        plt.arrow(-pos_1[2], pos_1[0], arrow_1_dx, arrow_1_dy, color="red")
        plt.arrow(-pos_2[2], pos_2[0], arrow_2_dx, arrow_2_dy, color="red")
        # Get arrows to make axis
        pitch_f = 0
        #        arrow_1a_dx = 1.5 * np.sin(pitch_f + np.pi / 2)
        #        arrow_1a_dy = 1.5 * np.cos(pitch_f + np.pi / 2)
        #        arrow_1b_dx = 1.5 * np.sin(pitch_f + np.pi)
        #        arrow_1b_dy = 1.5 * np.cos(pitch_f + np.pi)
        #        arrow_1c_dx = 1.5 * np.sin(pitch_f + 3 * np.pi / 2)
        #        arrow_1c_dy = 1.5 * np.cos(pitch_f + 3 * np.pi / 2)
        #        arrow_1d_dx = 1.5 * np.sin(pitch_f + 4 * np.pi / 2)
        #        arrow_1d_dy = 1.5 * np.cos(pitch_f + 4 * np.pi / 2)
        #        plt.arrow(-pos_1[2], pos_1[0], arrow_1a_dx, arrow_1a_dy)
        #        plt.arrow(-pos_1[2], pos_1[0], arrow_1b_dx, arrow_1b_dy)
        #        plt.arrow(-pos_1[2], pos_1[0], arrow_1c_dx, arrow_1c_dy)
        #        plt.arrow(-pos_1[2], pos_1[0], arrow_1d_dx, arrow_1d_dy)
        # make arrows for heading frame
        arrow_1a_dx = 1.5 * np.cos(pitch_1 + np.pi / 2)
        arrow_1a_dy = 1.5 * np.sin(pitch_1 + np.pi / 2)
        arrow_1b_dx = 1.5 * np.cos(pitch_1 + np.pi)
        arrow_1b_dy = 1.5 * np.sin(pitch_1 + np.pi)
        arrow_1c_dx = 1.5 * np.cos(pitch_1 + 3 * np.pi / 2)
        arrow_1c_dy = 1.5 * np.sin(pitch_1 + 3 * np.pi / 2)
        arrow_1d_dx = 1.5 * np.cos(pitch_1 + 4 * np.pi / 2)
        arrow_1d_dy = 1.5 * np.sin(pitch_1 + 4 * np.pi / 2)
        plt.arrow(-pos_1[2], pos_1[0], arrow_1a_dx, arrow_1a_dy)
        plt.arrow(-pos_1[2], pos_1[0], arrow_1b_dx, arrow_1b_dy)
        plt.arrow(-pos_1[2], pos_1[0], arrow_1c_dx, arrow_1c_dy)
        # plt.arrow(-pos_1[2], pos_1[0], arrow_1d_dx, arrow_1d_dy)
        # Draw arrow that goes towards goal from start
        plt.arrow(-pos_1[2], pos_1[0], -pos_2[2], pos_2[0], color="blue")
        plt.show()
        print("**********************************************")

    def dist_angle_to_indicies(self, dist, angle):
        angle = angle + np.pi
        angle_granularity = 2 * np.pi / self.angle_boxes
        i = int(angle / angle_granularity)
        distance_granularity = self.max_distance / self.distance_boxes
        j = int(dist / distance_granularity)
        return i, j

    def get_env_names(self):
        paths = glob.glob(self.dict_path + "*")
        paths.sort()
        names = [Path(Path(path).stem).stem for path in paths]
        return names

    def visualize_dict(self, d):
        start = d["start_position"]
        goal = d["goals"][0]["position"]
        traj = []
        traj.append(start)
        for pose in d["shortest_paths"][0]:
            traj.append(pose["position"])
        traj.append(goal)
        traj = np.array(traj)
        plt.plot(traj[:, 2], traj[:, 0])
        plt.savefig("./dict_visualization.jpg")

    def __len__(self):
        return len(self.dataset)

    def get_image(self, env, episode, location):
        start_time = time.time()
        image = plt.imread(
            self.image_data_path
            + env
            + "/"
            + "episodeRGB"
            + str(episode)
            + "_"
            + str(location).zfill(5)
            + ".jpg"
        )
        image = cv2.resize(image, (224, 224)) / 255
        depth = (
            plt.imread(
                self.image_data_path
                + env
                + "/"
                + "episodeDepth"
                + str(episode)
                + "_"
                + str(location).zfill(5)
                + ".jpg"
            )
            / 255
        )
        return image, depth

    # Images are offset by self.max_distance, because this should also detect going backwards which the robot can not do.
    def __getitem__(self, idx):
        env, sample_1, sample_2, distance, angle = self.dataset[idx]
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        transform_depth = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        if not self.debug:
            image1, depth1 = self.get_image(env, sample_1[0], sample_1[1])
            image2, depth2 = self.get_image(env, sample_2[0], sample_2[1])
            image1 = transform(image1)
            image2 = transform(image2)
            depth1 = transform_depth(depth1)
            depth2 = transform_depth(depth2)

        else:
            print(sample_1, sample_2)
            print(env)
            image1, depth1 = self.get_image(env, sample_1[0], sample_1[1])
            image2, depth2 = self.get_image(env, sample_2[0], sample_2[1])
        #            print(self.get_heading_diff(sample_1, sample_2, env))

        #             image1 = np.random.rand(224, 224, 3)
        #             image2 = np.random.rand(224, 224, 3)
        #             depth1 = np.random.rand(256, 256, 1)
        #             depth2 = np.random.rand(256, 256, 1)
        #        x = (image1, image2, depth1, depth2)
        x = (image1, image2)
        if sample_1 == sample_2:
            y = np.array([0.0])
        else:
            #            y = np.array([distance, angle])
            y = np.array([angle])
        return (x, y)

    def visualize_sample(self, x, y):
        # Requires debug to be on or wont work.
        assert self.debug
        images = []
        for image in x:
            temp_img = cv2.resize(image, (224, 224))
            if len(temp_img.shape) == 2:
                temp_img = np.stack([temp_img, temp_img, temp_img], axis=2)
            images.append(temp_img)
        im = np.hstack(images)
        plt.text(50, 25, str(y))
        plt.imshow(im)
        plt.show()

    def get_heading_diff(self, local_start, local_goal, env):
        d = self.labels[env]
        pos_start, rot_start = get_node_pose(local_start, d)
        pos_goal, rot_goal = get_node_pose(local_goal, d)
        return angle_between_quaternions(rot_start, rot_goal)

def get_displacement_label(local_start, local_goal, d):
    # See https://github.com/facebookresearch/habitat-lab/blob/b7a93bc493f7fb89e5bf30b40be204ff7b5570d7/habitat/tasks/nav/nav.py
    # for more information
    pos_start, rot_start = get_node_pose(local_start, d)
    pos_goal, rot_goal = get_node_pose(local_goal, d)
    # Quaternion is returned as list, need to change datatype
    direction_vector = pos_goal - pos_start
    direction_vector_start = quaternion_rotate_vector(
        rot_start.inverse(), direction_vector
    )
    rho, phi = cartesian_to_polar(-direction_vector_start[2], direction_vector_start[0])

    # Should be same as agent_world_angle
    norm_distance = np.linalg.norm(direction_vector)
    return np.array([norm_distance, rho, -phi])


def get_displacement_label_me(local_start, local_goal, d):
    # See https://github.com/facebookresearch/habitat-lab/blob/b7a93bc493f7fb89e5bf30b40be204ff7b5570d7/habitat/tasks/nav/nav.py
    # for more information
    pos_start, rot_start = get_node_pose(local_start, d)
    pos_goal, rot_goal = get_node_pose(local_goal, d)
    # Quaternion is returned as list, need to change datatype
    direction_vector = pos_goal - pos_start
    norm_distance = np.linalg.norm(direction_vector)
    direction_vector[1] = 0.0
    pitch = quaternion_to_yaw(rot_start)
    rot_start = quaternion.quaternion(*[0.0, 0.0, 0.0, 1.0])
    #    direction_vector_start = quaternion_rotate_vector(
    #        rot_start.inverse(), direction_vector
    #    )
    foo = rot_start.inverse()
    invert_me = quaternion.quaternion(
        *[rot_start.z, rot_start.y, rot_start.x, rot_start.w]
    )
    foo = invert_me.inverse()
    bar = quaternion.quaternion(*[foo.x, foo.y, foo.z, foo.w])
    direction_vector_start = quaternion.as_rotation_matrix(invert_me) @ direction_vector
    direction_vector_start_2 = quaternion.as_rotation_matrix(foo) @ direction_vector
    # direction_vector_start = quaternion_rotate_vector(bar, direction_vector)
    # print(direction_vector_start)
    rho, phi = cartesian_to_polar(-direction_vector_start[2], direction_vector_start[0])
    phi -= pitch
    # Should be same as agent_world_angle
    if phi < -np.pi:
        phi = (2 * np.pi) + phi
    elif phi > np.pi:
        phi = (2 * np.pi) - phi
    return np.array([norm_distance, rho, phi])


def get_node_pose(node, d):
    pose = d[node[0]]["shortest_paths"][0][0][node[1]]
    position = pose["position"]
    rotation = pose["rotation"]
    return np.array(position),quaternion_from_coeff(rotation)


def quaternion_to_yaw(q):
    d1 = np.array([0, 0, -1])
    d2 = quaternion.as_rotation_matrix(q) @ d1
    v1 = np.array([-d1[2], d1[0]])
    v2 = np.array([-d2[2], d2[0]])
    # https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
    ang = np.arctan2(v2[0] * v1[1] - v1[0] * v2[1], v2[0] * v1[0] + v2[1] * v1[1])
    return ang


#    return np.arccos(
#        np.clip(np.dot(direction_vector, direction_vector_rotated), -1.0, 1.0)
#    )


def approx_equal(x, y, d=0.000001):
    if np.abs(x - y) < d:
        return 1
    else:
        return 0


def test_quat_stuff():
    q = quaternion.quaternion(*[0.0, 0.0, 0.0, 1.0])
    assert approx_equal(quaternion_to_yaw(q), 0.0)
    q = quaternion.quaternion(*[0.0, 0.0, 0.0, -1.0])
    assert approx_equal(quaternion_to_yaw(q), 0.0)
    q = quaternion.quaternion(*[0.0, 1.0, 0.0, 0.0])
    assert approx_equal(quaternion_to_yaw(q), math.radians(180)) or approx_equal(
        quaternion_to_yaw(q), math.radians(-180)
    )
    q = quaternion.quaternion(*[0.0, 0.707, 0.0, 0.707])
    assert approx_equal(quaternion_to_yaw(q), math.radians(90))
    q = quaternion.quaternion(*[0.0, 0.707, 0.0, -0.707])
    assert approx_equal(quaternion_to_yaw(q), math.radians(-90))
    q = quaternion.quaternion(*[0.0, -0.707, 0.0, -0.707])
    assert approx_equal(quaternion_to_yaw(q), math.radians(90))
    q = quaternion.quaternion(*[0.0, -0.707, 0.0, 0.707])
    assert approx_equal(quaternion_to_yaw(q), math.radians(-90))


if __name__ == "__main__":
    test_quat_stuff()
    dataset = GibsonDataset(
        "train",
        samples=2000,
        seed=0,
        angle_boxes=12,
        distance_boxes=15,
        max_distance=1.5,
        ignore_0=False,
        debug=True,
        episodes=20,
    )
    max_angle = 0
    max_displacement = 0
    displacements = []
    angles = []
    ys = []
    for batch in tqdm(dataset):
        (
            x,
            y,
        ) = batch
        dataset.visualize_sample(x, y)
    #        im1, im2 = x
    #        distance, angle = y
    #        displacements.append(distance)
    #        angles.append(angle)
    print(max_angle, max_displacement)
    plt.hist(displacements, bins=1000)
    plt.show()
    plt.clf()
    plt.hist(angles, bins=1000)
    plt.show()
    plt.clf()
