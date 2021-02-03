# Basically need to do the following
# 1) Get trajectories and labels (Done)
# 2) Take model from before, but use regression
# 3) Data should also be similiar to before

import argparse
from tqdm import tqdm
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


def train(model, device, epochs=30):
    current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    results_dir = "./data/results/localization/" + current_time + "/"
    os.mkdir(results_dir)
    copyfile("./model.py", results_dir + "model.py")
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    BATCH_SIZE = 64
    seed = 0
    train_dataset = GibsonDataset(
        "train",
        seed,
        samples=100,
        max_distance=30,
        episodes=20,
        ignore_0=False,
        debug=False,
    )
    #    train_dataset.save_env_data(results_dir)
    train_dataloader = data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=16,
    )
    test_dataset = GibsonDataset(
        "test",
        seed,
        samples=10,
        max_distance=30,
        episodes=20,
        ignore_0=False,
        debug=False,
    )
    test_dataloader = data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=16,
    )
    train_acc = []
    losses = []
    losses_cum = []
    val_acc = []
    losses_v = []
    losses_v_cum = []
    for epoch in range(epochs):
        print("epoch ", epoch)
        model.train()
        for i, batch in enumerate(tqdm(train_dataloader)):
            x, y = batch
            im1, im2, depth1, depth2 = x
            y = y.type(torch.float32)
            y = y.to(device)
            im1 = im1.to(device).float()
            im2 = im2.to(device).float()
            depth1 = depth1.to(device).float()
            depth2 = depth2.to(device).float()
            out = model(im1, im2, depth1, depth2)
            optimizer.zero_grad()
            loss = criterion(out, y)
            loss.backward()
            loss = loss.cpu().detach().numpy()
            losses.append(loss)
            optimizer.step()

        losses_cum.append(np.mean(losses))
        print(np.mean(losses))
        losses = []
        # Do validation
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(
                track(test_dataloader, description="[red] Testing!")
            ):
                x, y = batch
                im1, im2, depth1, depth2 = x
                y = y.type(torch.float32)
                y = y.to(device)
                im1 = im1.to(device).float()
                im2 = im2.to(device).float()
                depth1 = depth1.to(device).float()
                depth2 = depth2.to(device).float()
                out = model(im1, im2, depth1, depth2)
                loss = criterion(out, y)
                loss = loss.cpu().detach().numpy()
                losses_v.append(loss)
        losses_v_cum.append(np.mean(losses_v))
        print(np.mean(losses_v))
        losses_v = []
        if epoch % 5 == 0:
            torch.save(model.state_dict(), results_dir + "saved_model.pth")
    torch.save(model.state_dict(), results_dir + "saved_model.pth")
    np.save(results_dir + "losses.npy", losses_cum)
    np.save(results_dir + "losses_v.npy", losses_v_cum)
    plt.plot(losses_v_cum, label="Test Loss")
    plt.plot(losses_cum, label="Train Loss")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(results_dir + "loss.jpg")
    print(current_time)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--network_location", help="No", required=False)
    args = parser.parse_args()
    device = torch.device("cuda:0")
    model = Siamese().to(device)
    if args.network_location:
        model.load_state_dict(torch.load(args.network_location))
    model = train(model, device)
