import glob
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
        give_distance=False,
    ):
        random.seed(seed)
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
        pu.db
        if visualize:
            d = get_dict(self.train_env[0])
            self.visualize_dict(d[0])
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

    #        self.verify_dataset()

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

    def balance_dataset(self, max_number_of_examples=2000):
        min_size = np.inf
        if self.ignore_0:
            dist_range = range(1, self.max_distance)
        else:
            dist_range = range(self.max_distance)
        for i in dist_range:
            min_size = min(min_size, len(self.dataset[i]))
        min_size = min(min_size, max_number_of_examples)
        far_away_keys = range(self.max_distance, max(self.dataset.keys()))
        ret = {}
        far_away_dataset = []
        for i in dist_range:
            weight = 1.0
            ret[i] = random.sample(self.dataset[i], int(min_size * weight))
        # last entrie is gonna be far away examples
        ret[self.max_distance] = []
        for i in range(min_size):
            random_far_away = random.choice(far_away_keys)
            ret[self.max_distance].append(random.choice(self.dataset[random_far_away]))
        return ret

    # Don't forget map/trajectory is directed.
    def get_dataset(self, envs):
        ret = {}
        retries = 10
        if self.debug:
            envs = envs[0:3]
        for distance in tqdm(range(400)):
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
                    # Get images in the forward direction
                    if distance < 0:
                        for i, j in zip(
                            range(-distance, path_length), range(path_length)
                        ):
                            possible_ij.append((i, j))
                    else:
                        for i, j in zip(
                            range(path_length), range(distance, path_length)
                        ):
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
        image1 = self.get_image(env, episode, l1)
        image2 = self.get_image(env, episode, l2)
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
        y = l2 - l1
        if y >= int(self.max_distance / 2) or y < 0:
            y = 0
        else:
            y = 1
        if self.give_distance:
            return (x, y, l2 - l1)
        return (x, y)


if __name__ == "__main__":
    dataset = GibsonDataset(
        "train",
        samples=10,
        seed=0,
        max_distance=60,
        ignore_0=False,
        debug=True,
        episodes=20,
        give_distance=True
    )
    max_angle = 0
    max_displacement = 0
    displacements = []
    angles = []
    ys = []
    pu.db
    for batch in tqdm(dataset):
        (
            x,
            y,
            d
        ) = batch
        #        dataset.visualize_sample(x, y, episode, l1, l2)
        im1, im2 = x
        ys.append(d)
        # im = np.hstack([im1, im2])
        # plt.text(50, 25, str(y))
        # plt.imshow(im)
        # plt.show()
    plt.hist(ys, bins=1000)
    plt.savefig("histogram_y.jpg")
