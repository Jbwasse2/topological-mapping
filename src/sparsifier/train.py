import argparse
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


def train(model, device, epochs=200):
    print("DONT FORGET TO UNCOMMENT SAVE MODEL")
    current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    results_dir = "../../data/results/sparsifier/" + current_time + "/"
    os.mkdir(results_dir)
    copyfile("./model.py", results_dir + "model.py")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    BATCH_SIZE = 64
    seed = 0
    train_dataset = GibsonDataset("train", seed, samples=30000, max_distance=5, ignore_0=True)
    train_dataset.save_env_data(results_dir)
    train_dataloader = data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=16,
    )
    test_dataset = GibsonDataset("test", seed, samples=3500, max_distance=5, ignore_0=True)
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
    accuracy = []
    accuracy_cum = []
    accuracy_v = []
    accuracy_v_cum = []
    for epoch in range(epochs):
        print("epoch ", epoch)
        model.train()
        for i, batch in enumerate(track(train_dataloader, description="[cyan] Training!")):
            x, y = batch
            im1, im2 = x
            y = y.to(device)
            im1 = im1.to(device).float()
            im2 = im2.to(device).float()
            out = model(im1, im2)
            optimizer.zero_grad()
            loss = criterion(out, y)
            loss.backward()
            loss = loss.cpu().detach().numpy()
            losses.append(loss)
            optimizer.step()
            choose = np.argmax(out.cpu().detach().numpy(), axis=1)
            gt = y.cpu().detach().numpy()
            accuracy.append(len(np.where(gt == choose)[0]) / len(gt))

        losses_cum.append(np.mean(losses))
        accuracy_cum.append(np.mean(accuracy))
        print(np.mean(losses))
        print(np.mean(accuracy))
        losses = []
        accuracy = []
        # Do validation
        model.eval()
        for i, batch in enumerate(track(test_dataloader, description="[red] Testing!")):
            x, y = batch
            im1, im2 = x
            y = y.to(device)
            im1 = im1.to(device).float()
            im2 = im2.to(device).float()
            out = model(im1, im2)
            loss = criterion(out, y)
            loss.backward()
            loss = loss.cpu().detach().numpy()
            losses_v.append(loss)
            choose = np.argmax(out.cpu().detach().numpy(), axis=1)
            gt = y.cpu().detach().numpy()
            accuracy_v.append(len(np.where(gt == choose)[0]) / len(gt))
        losses_v_cum.append(np.mean(losses_v))
        accuracy_v_cum.append(np.mean(accuracy_v))
        print(np.mean(losses_v))
        print(np.mean(accuracy_v))
        losses_v = []
        accuracy_v = []
        if epoch % 5 == 0:
            torch.save(model.state_dict(), results_dir + "saved_model.pth")
    torch.save(model.state_dict(), results_dir + "saved_model.pth")
    np.save(results_dir + "losses.npy", losses_cum)
    np.save(results_dir + "losses_v.npy", losses_v_cum)
    np.save(results_dir + "accuracy.npy", accuracy_cum)
    np.save(results_dir + "accuracy_v.npy", accuracy_v_cum)
    print(results_dir)
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
