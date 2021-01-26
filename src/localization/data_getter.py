import glob
import quaternion
import gzip
import random
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pudb
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision import transforms as transforms
from torchvision import utils
from torchvision.transforms import ToTensor
from rich.progress import track

from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import angle_between_quaternions


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
        visualize=False,
        episodes=100,
        max_distance=10,
        samples=10000,
        ignore_0=False,
        debug=False,
    ):
        random.seed(seed)
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
        if self.ignore_0:
            self.dataset.pop(0, None)
        self.dataset = self.flatten_dataset()
        self.verify_dataset()

    def get_labels_dicts(self, envs):
        if self.debug:
            envs = envs[0:3]
        ret = {}
        self.distances = []
        self.angles = []
        for env in track(envs, description="[green] Collecting Labels"):
            ret[env] = get_dict(env)
            # Get distance/rotation
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
            key = data[3] - data[2]
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
        for key, value in self.dataset.items():
            ret = ret + value
        random.shuffle(ret)
        return ret

    def balance_dataset(self, max_number_of_examples=2000):
        min_size = np.inf
        if self.ignore_0:
            dist_range = range(1, self.max_distance + 1)
        else:
            dist_range = range(self.max_distance + 1)
        for i in dist_range:
            min_size = min(min_size, len(self.dataset[i]))
        min_size = min(min_size, max_number_of_examples)
        far_away_keys = range(self.max_distance, max(self.dataset.keys()))
        ret = {}
        far_away_dataset = []
        for i in dist_range:
            ret[i] = random.sample(self.dataset[i], min_size)
        return ret

    # Don't forget map/trajectory is directed.
    def get_dataset(self, envs):
        ret = {}
        retries = 100
        if self.debug:
            envs = envs[0:3]
        for distance in track(
            range(self.max_distance + 1), description="[green] Collecting Data"
        ):
            for sample in range(self.samples):
                # Keep trying to find sample that satisfies requirements...
                for _ in range(retries):
                    # Choose a random env
                    env = random.choice(envs)
                    # Choose random episode in env
                    episode = random.choice(range(len(self.labels[env])))
                    # Get all pairs of i,j that satisfy distance
                    path_length = len(self.labels[env][episode]["shortest_paths"][0][0])
                    possible_ij = []
                    for i, j in zip(range(path_length), range(distance, path_length)):
                        possible_ij.append((i, j))
                    # If sample is found, break
                    if len(possible_ij) != 0:
                        i, j = random.choice(possible_ij)
                        key = j - i
                        if key not in ret:
                            ret[key] = []
                        # Ignore if distance between nodes is 0, this causes problems...
                        pos1 = self.labels[env][episode]["shortest_paths"][0][0][i][
                            "position"
                        ]
                        pos2 = self.labels[env][episode]["shortest_paths"][0][0][j][
                            "position"
                        ]
                        if (pos1 != pos2 and i != j) or i == j:
                            ret[key].append((env, episode, i, j))
                            break
        return ret

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
        return image

    # Images are offset by self.max_distance, because this should also detect going backwards which the robot can not do.
    def __getitem__(self, idx):
        env, episode, l1, l2 = self.dataset[idx]
        #    image1 = self.get_image(env, episode, l1)
        #    image2 = self.get_image(env, episode, l2)
        #    transform = transforms.Compose(
        #        [
        #            transforms.ToTensor(),
        #            transforms.Normalize(
        #                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        #            ),
        #        ]
        #    )
        #    image1 = transform(image1)
        #    image2 = transform(image2)

        #    x = (image1, image2)
        x = None
        d = self.labels[env]
        if l1 == l2:
            y = np.array([0.0, 0.0])
        else:
            y = get_displacement_label((episode, l1), (episode, l2), d)
        return (x, y)

    def visualize_sample(self, x, y, episode, l1, l2):
        # Requires debug to be on or wont work.
        assert self.debug
        im1, im2 = x
        im = np.hstack([im1, im2])
        plt.text(50, 25, str(y) + "_" + str(episode) + "_" + str(l1) + "_" + str(l2))
        plt.imshow(im)
        plt.show()


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
    return np.array([rho, -phi])


def get_node_pose(node, d):
    pose = d[node[0]]["shortest_paths"][0][0][node[1]]
    position = pose["position"]
    rotation = pose["rotation"]
    return np.array(position), quaternion.quaternion(*rotation)


if __name__ == "__main__":
    dataset = GibsonDataset(
        "train",
        samples=2000,
        seed=0,
        max_distance=30,
        ignore_0=False,
        debug=False,
        episodes=20,
    )
    max_angle = 0
    max_displacement = 0
    displacements = []
    angles = []
    for batch in tqdm(dataset):
        (
            x,
            y,
        ) = batch
        #    dataset.visualize_sample(x, y, episode, l1, l2)
        #        im1, im2 = x
        y = y
        max_angle = max(y[1], max_angle)
        max_displacement = max(y[0], max_displacement)
        displacements.append(y[0])
        angles.append(y[1])
    print(max_angle, max_displacement)
    plt.hist(displacements, bins=999)
    plt.show()
    plt.clf()
    plt.hist(angles, bins=1000)
    plt.show()
    np.save("./disp.npy", displacements)
    np.save("./angles.npy", angles)
