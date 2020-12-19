import glob
import gzip
import random
import re
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pudb
import torch
import torchvision.transforms as transforms
from natsort import os_sorted
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as transforms
from torchvision import utils
from torchvision.transforms import ToTensor
from tqdm import tqdm

matplotlib.use("Agg")


class GibsonMapDataset(Dataset):
    def __init__(self, test_envs, transform=True):
        self.transform = transform
        self.image_data_path = (
            "../../data/datasets/pointnav/gibson/v2/train_large/images/"
        )
        self.dict_path = "../../data/datasets/pointnav/gibson/v2/train_large/content/"
        self.test_env = test_envs
        self.current_env = None

    def set_env(self, env):
        assert env in self.test_env
        self.current_env = env
        self.dataset = glob.glob(self.image_data_path + self.current_env + "/*")
        self.dataset = os_sorted(self.dataset)
        self.last = self.last_image_in_episode(self.dataset)
        self.number_of_trajectories = np.sum(self.last)

    def get_image(self, path):
        image = plt.imread(path)
        return image

    def last_image_in_episode(self, dataset):
        ret = []
        for i in range(len(dataset)):
            if i + 1 == len(dataset):
                ret.append(1)
                continue
            s1 = re.search(r"episode\d+", dataset[i]).group()
            s2 = re.search(r"episode\d+", dataset[i + 1]).group()
            if s1 == s2:
                ret.append(0)
            else:
                ret.append(1)
        assert len(ret) == len(dataset)
        return ret

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.get_image(self.dataset[idx])
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        if self.transform:
            image = cv2.resize(image, (224, 224)) / 255
            image = transform(image)
        return image, self.last[idx]


if __name__ == "__main__":
    data = GibsonMapDataset(["Airport", "Azusa"])
    data.set_env("Airport")
    print(len(data))
    data[0]
    data.set_env("Azusa")
    print(len(data))
    data[0]
