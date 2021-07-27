import glob
import time
import os
import gzip
import random
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

from rich.progress import track
from habitat.utils.geometry_utils import angle_between_quaternions

matplotlib.use("Agg")


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
        episodes=20,
        max_distance=10,
        samples=10000,
        ignore_0=False,
        debug=False,
        context=10,
        give_distance=False,
    ):
        random.seed(seed)
        self.context = context
        self.give_distance = give_distance
        self.debug = debug
        self.ignore_0 = ignore_0
        self.split_type = split_type
        self.max_distance = max_distance
        # This is changed in the train_large_data.py file
        self.episodes = episodes
        self.samples = samples
        self.image_data_path = (
            "../../data/datasets/pointnav/gibson/v4/train_large/images/"
        )
        self.dict_path = "../../data/datasets/pointnav/gibson/v4/train_large/content/"
        self.env_names = self.get_env_names()
        random.shuffle(self.env_names)
        split = 0.85
        self.train_env = self.env_names[0 : int(len(self.env_names) * split)]
        self.test_env = self.env_names[int(len(self.env_names) * split) :]
        if visualize:
            d = get_dict(self.train_env[-1])
            self.visualize_dict(d[0])
        if split_type == "train":
            self.labels = self.get_labels_dicts(self.train_env)
            self.number_of_images_in_envs = self.get_sequence_lengths(self.train_env)
            self.dataset = self.get_dataset(self.train_env)
        elif split_type == "test":
            self.labels = self.get_labels_dicts(self.test_env)
            self.number_of_images_in_envs = self.get_sequence_lengths(self.test_env)
            self.dataset = self.get_dataset(self.test_env)

    def get_sequence_lengths(self,envs):
        # Get number of images in envs for each episode
        if self.debug:
            envs = envs[0:3]
        number_of_images_in_envs = {}
        start = time.clock() 
        for env in tqdm(envs):
            number_of_images_in_envs[env] = {}
            for e in range(self.episodes):
                number_of_images_in_envs[env][e] = len(glob.glob(self.image_data_path + env + "/episodeRGB" + str(e) + "_*.jpg"))
        # This dataset does start from 0 unlike other code, but context creates an offset.
        return number_of_images_in_envs

    def get_labels_dicts(self, envs):
        if self.debug:
            envs = envs[0:3]
        ret = {}
        self.distances = []
        self.angles = []
        for env in tqdm(envs):
            ret[env] = get_dict(env)
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
            key = np.abs(data[2] - data[3])
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


    # Don't forget map/trajectory is directed.
    # diff_traj_split controls what percent of negative samples are in
    def get_dataset(self, envs, diff_traj_split=0.5):
        if self.debug:
            envs = envs[0:3]
        image_offset_in_envs = {}
        for env in envs:
            image_offset_in_envs[env] = self.context
        # Would like to get a dataset that is equally weighted between
        # All possible lengths in max_distance. So for example if max_dist=10
        # and samples = 1000, we'd want 500 total samples <= 10 dist with 50 samples
        # in 0...10.
        # As per the ViNG paper, the last 500 samples should be a mix of same traj
        # and other traj. Let's try 50% 50%? This is called the diff_traj_split

        # Get positive samples
        label = 1
        dataset = []
        positive_samples_per_distance = int(
            (self.samples / 2) / (self.max_distance + 1)
        )
        for distance in tqdm(range(self.max_distance + 1)):
            for _ in range(positive_samples_per_distance):
                # Choose random env uniformly
                env = random.choice(envs)
                episode = random.choice(range(self.episodes))
                sample_range = range(
                        image_offset_in_envs[env],
                        -image_offset_in_envs[env]
                        + self.number_of_images_in_envs[env][episode]
                        - distance,
                    )
                if len(sample_range) == 0:
                    continue
                traj_local_start = random.choice(sample_range)
                traj_local_end = traj_local_start + distance
                dataset.append((env, env, traj_local_start, traj_local_end, label, episode, episode))

        # Get negative samples in same trajectory
        label = 0
        neg_samples_in_same_traj = int((self.samples / 2) * (1 - diff_traj_split))
        for _ in tqdm(range(neg_samples_in_same_traj)):
            # We need to make sure randomly selected data is not accidently positive
            while 1:
                env = random.choice(envs)
                episode = random.choice(range(self.episodes))
                sample_range = range(
                        image_offset_in_envs[env],
                        -image_offset_in_envs[env] + self.number_of_images_in_envs[env][episode],
                    )
                if len(sample_range) == 0:
                    continue
                location_1, location_2 = random.sample(sample_range, 2)
                start = min(location_1, location_2)
                end = max(location_1, location_2)
                if np.abs(start - end) > self.max_distance:
                    dataset.append((env, env, start, end, label, episode, episode))
                    break

        # Get negative samples in different trajectories
        neg_samples_in_diff_traj = int((self.samples / 2) * (diff_traj_split))
        for _ in tqdm(range(neg_samples_in_diff_traj)):
            if len(envs) > 2:
                env1, env2 = random.sample(envs, 2)
            else:
                env1 = envs[0]
                env2 = envs[0]
            episode1 = random.choice(range(self.episodes))
            episode2 = random.choice(range(self.episodes))
            start_range =  range(
                    image_offset_in_envs[env1],
                    -image_offset_in_envs[env1] + self.number_of_images_in_envs[env1][episode1],
                )
            if len(start_range) == 0:
                continue
            start = random.choice(start_range)
            end_range = range(
                    image_offset_in_envs[env2],
                    -image_offset_in_envs[env2] + self.number_of_images_in_envs[env2][episode2],
                )
            if len(end_range) == 0:
                continue
            end = random.choice(end_range)
            dataset.append((env1, env2, start, end, label, episode1, episode2))
        return dataset

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

    def get_images(self, env, episode, location):
        ret = []
        for loc in range(location - self.context, location):
            location = (self.image_data_path
                + env
                + "/"
                + "episodeRGB"
                + str(episode)
                + "_"
                + str(loc).zfill(5)
                + ".jpg")
            image = plt.imread(location)
            image = cv2.resize(image, (224, 224)) / 255
            ret.append(image)
        return ret

    # Images are offset by self.max_distance, because this should also detect going backwards which the robot can not do.
    def __getitem__(self, idx):
        env1, env2, l1, l2, y, ep1, ep2 = self.dataset[idx]
        seq1 = self.get_images(env1, ep1, l1)
        seq2 = self.get_images(env2, ep2, l2)
        pose1 = self.labels[env1][ep1]['shortest_paths'][0][0][l1]
        pose2 = self.labels[env2][ep2]['shortest_paths'][0][0][l2]
        positionDiff = (np.array(pose1['position']) - np.array(pose2['position']))
        rotDiff = angle_between_quaternions(np.quaternion(*pose1['rotation']), np.quaternion(*pose2['rotation']))
        poseDiff = np.append(positionDiff, rotDiff)
        pose1 = pose1['position'] + pose1['rotation']
        pose2 = pose2['position'] + pose2['rotation']
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        for count, image in enumerate(seq1):
            seq1[count] = transform(image)
        for count, image in enumerate(seq2):
            seq2[count] = transform(image)
        seq1 = np.stack(seq1)
        seq2 = np.stack(seq2)

        x = (seq1, seq2)
        y = l2 - l1
        if y >= int(self.max_distance / 2) or y < 0:
            y = 0
        else:
            y = 1
        if self.give_distance:
            ys = l2 - l1
            if env1 != env2:
                ys = -1
            return (x, y, ys, poseDiff)
        return (x, y, poseDiff)


if __name__ == "__main__":
    train_dataset = GibsonDataset(
        "train",
        0,
        samples=2000,
        max_distance=30,
        episodes=20,
        ignore_0=False,
        debug=True,
    )
    max_angle = 0
    max_displacement = 0
    displacements = []
    angles = []
    ys = []
    for batch in tqdm(train_dataset):
        (
            x,
            y1,
            y2
        ) = batch
        #        dataset.visualize_sample(x, y, episode, l1, l2)
        ys.append(y2[3])
        # im = np.hstack([im1, im2])
        # plt.text(50, 25, str(y))
        # plt.imshow(im)
        # plt.show()
    plt.hist(ys, bins=1000)
    plt.savefig("histogram_y.jpg")
