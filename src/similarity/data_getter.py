import glob
import re
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


# Image dataset for training ViNG similarity detector
# Inputs:
# -split_type: Train or Test
# -seed: seed #
# -max_distance: maximum time steps between images in trajectory to be
# considered close.
# -samples: Total number of samples to collect
# -debug: True or False. If true collects a smaller dataset for
# debugging. If False class behaves normally
# -give_tuple: when _getitem_ is called it will also return the tuple that was used
# this is useful for debbuging
class ViNGImageDataset(Dataset):
    """ Dataset collect from Habitat """

    def __init__(
        self,
        split_type,
        image_data_path,
        seed=0,
        max_distance=10,
        samples=10000,
        give_distance=False,
        debug=False,
    ):
        random.seed(seed)
        self.debug = debug
        self.give_distance = give_distance
        self.split_type = split_type
        self.max_distance = max_distance
        # This is changed in the train_large_data.py file
        self.samples = samples
        self.image_data_path = image_data_path
        self.env_names = self.get_env_names()
        random.shuffle(self.env_names)
        split = 0.85
        self.train_env = self.env_names[0 : int(len(self.env_names) * split)]
        self.test_env = self.env_names[int(len(self.env_names) * split) :]
        if split_type == "train":
            self.dataset = self.get_dataset(self.train_env)
        elif split_type == "test":
            self.dataset = self.get_dataset(self.test_env)

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

    # diff_traj_split controls what percent of negative samples are in
    # different trajectories
    # Dataset structure is a list of tuples that contain the following
    # (env_start, env_end, start, end, label)
    # The env for the start and end may be different as in ViNG, but this
    # is only the case for the negative samples case.
    def get_dataset(self, envs, diff_traj_split=0.50):
        # Get number of images in envs
        number_of_images_in_envs = {}
        for env in envs:
            number_of_images_in_envs[env] = len(glob.glob(env + "/*"))
        # The dataset numbering may not start from 0...
        image_offset_in_envs = {}
        for env in envs:
            image_offset = glob.glob(env + "/*")
            image_offset.sort()
            image_offset_lowest = image_offset[0]
            image_name = os.path.basename(image_offset_lowest)
            image_name_no_ext = os.path.splitext(image_name)[0]
            image_offset_in_envs[env] = int(image_name_no_ext)
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
                traj_local_start = random.choice(
                    range(
                        image_offset_in_envs[env],
                        image_offset_in_envs[env]
                        + number_of_images_in_envs[env]
                        - distance,
                    )
                )
                traj_local_end = traj_local_start + distance
                dataset.append((env, env, traj_local_start, traj_local_end, label))

        # Get negative samples in same trajectory
        label = 0
        neg_samples_in_same_traj = int((self.samples / 2) * (1 - diff_traj_split))
        for _ in tqdm(range(neg_samples_in_same_traj)):
            # We need to make sure randomly selected data is not accidently positive
            while 1:
                env = random.choice(envs)
                location_1, location_2 = random.sample(
                    range(
                        image_offset_in_envs[env],
                        image_offset_in_envs[env] + number_of_images_in_envs[env],
                    ),
                    2,
                )
                start = min(location_1, location_2)
                end = max(location_1, location_2)
                if np.abs(start - end) > self.max_distance:
                    dataset.append((env, env, start, end, label))
                    break

        # Get negative samples in different trajectories
        neg_samples_in_diff_traj = int((self.samples / 2) * (diff_traj_split))
        for _ in tqdm(range(neg_samples_in_diff_traj)):
            env1, env2 = random.sample(envs, 2)
            start = random.choice(
                range(
                    image_offset_in_envs[env1],
                    image_offset_in_envs[env1] + number_of_images_in_envs[env1],
                )
            )
            end = random.choice(
                range(
                    image_offset_in_envs[env2],
                    image_offset_in_envs[env2] + number_of_images_in_envs[env2],
                )
            )
            dataset.append((env1, env2, start, end, label))
        return dataset

    # This function recursively goes over the root image folder.
    # It will then go into each directory in the given directory and return all directories
    # that have images in them.
    def get_env_names(self):
        # Add env only if
        # 1) Has no subdirs
        # 2) First file in files list is jpg
        envs = []
        for root, subdirs, files in os.walk(self.image_data_path):
            if len(subdirs) == 0 and len(files) > 0:
                if ".jpg" in files[0]:
                    envs.append(root)
        return envs

    def __len__(self):
        return len(self.dataset)

    def get_image(self, env, frame):
        image = plt.imread(env + "/" + str(frame).zfill(7) + ".jpg")
        image = cv2.resize(image, (224, 224)) / 255
        return image

    # Images are offset by self.max_distance, because this should also detect going backwards which the robot can not do.
    def __getitem__(self, idx):
        env_start, env_end, start, end, label = self.dataset[idx]
        image1 = self.get_image(env_start, start)
        image2 = self.get_image(env_end, end)
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        image1 = transform(image1)
        image2 = transform(image2)

        x = (image1, image2)
        if self.give_distance:
            return (x, label, self.dataset[idx])
        return (x, label)


if __name__ == "__main__":
    dataset = ViNGImageDataset(
        "train",
        samples=10000,
        image_data_path="./data/clean/",
        seed=0,
        max_distance=20,
        debug=True,
        give_distance=True,
    )
    displacements = []
    ys = {}
    ys[0] = 0
    ys[1] = 0
    for batch in tqdm(dataset):
        (x, y, tup) = batch
        im1, im2 = x
        d = np.abs(tup[3] - tup[2])
        ys[y] += 1
        if y == 1:
            displacements.append(d)
    plt.hist(displacements, bins=100)
    plt.show()
    print(ys)
