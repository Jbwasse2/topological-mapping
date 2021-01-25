import argparse
import matplotlib.pyplot as plt

import os
import time
from shutil import copyfile

import numpy as np
import pudb
import torch
import torch.optim as optim
from torch import nn
from torch.utils import data
from rich.progress import track

from data_getter import GibsonDataset
from model import Siamese

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def evaluate(model, device):
    seed = 0
    BATCH_SIZE = 1
    model.eval()
    test_dataset = GibsonDataset(
        "test", seed, samples=2000, max_distance=10, episodes=10, ignore_0=False
    )
    criterion = nn.MSELoss()
    test_dataloader = data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=1,
    )
    losses_v = []
    with torch.no_grad():
        #    for i, batch in enumerate(track(test_dataloader, description="[red] Testing!")):
        for i, batch in enumerate(test_dataloader):
            x, y = batch
            im1, im2 = x
            y = y.type(torch.float32)
            y = y.to(device)
            im1 = im1.to(device).float()
            im2 = im2.to(device).float()
            out = model(im1, im2)
            loss = criterion(out, y)
            pu.db
            loss = loss.cpu().detach().numpy()
            im1, im2 = x
            visualize(test_dataset, i, out, y, im1, im2)
            losses_v.append(loss)
        losses_v_cum.append(np.mean(losses_v))
    print(np.mean(losses_v))


def visualize(test_dataset, i, out, y, im1, im2):
    (env, episode, i, j) = test_dataset.dataset[i]
    #    im1 = test_dataset.get_image(env, episode, i)
    #    im2 = test_dataset.get_image(env, episode, j)
    pu.db
    im1 = im1.squeeze().permute(1, 2, 0).numpy()
    im2 = im2.squeeze().permute(1, 2, 0).numpy()
    im = np.hstack([im1, im2])
    plt.text(
        25,
        25,
        str(
            "true = " + str(y[0].cpu().numpy()) + " pred = " + str(out[0].cpu().numpy())
        ),
    )
    plt.imshow(im)
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda:0")
    model = Siamese().to(device)
    model.load_state_dict(
        torch.load("./data/results/localization/best_model/saved_model.pth")
    )
    model = evaluate(model, device)
