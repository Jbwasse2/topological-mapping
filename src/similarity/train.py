import argparse
import os
import time
from shutil import copyfile

import numpy as np
import pudb
import torch
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from data_getter import ViNGImageDataset
from model import Siamese
from rich.progress import track

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def soft_classification_loss(guess, truth):
    guess_class = guess.argmax(1)
    return torch.abs(guess_class - truth).sum().float()


def train(model, device, epochs=30):
    current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    results_dir = "./data/results/similarity/" + current_time + "/"
    os.mkdir(results_dir)
    copyfile("./model.py", results_dir + "model.py")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    BATCH_SIZE = 64
    seed = 0
    train_dataset = ViNGImageDataset(
        "train",
        samples=10000,
        image_data_path="./data/clean/",
        seed=0,
        max_distance=20,
    )
    train_dataset.save_env_data(results_dir)
    train_dataloader = data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=16,
    )
    pu.db
    test_dataset = ViNGImageDataset(
        "test",
        samples=1000,
        image_data_path="./data/clean/",
        seed=0,
        max_distance=20,
    )
    test_dataloader = data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=16,
    )
    train_acc = []
    losses = []
    losses_cumulative = []
    val_acc = []
    losses_v = []
    losses_v_cumulative = []
    accuracy = []
    accuracy_cumulative = []
    accuracy_v = []
    accuracy_v_cumulative = []
    for epoch in range(epochs):
        print("epoch ", epoch)
        model.train()
        for i, batch in enumerate(tqdm(train_dataloader)):
            x, y = batch
            im1, im2 = x
            y = y.to(device)
            im1 = im1.to(device).float()
            im2 = im2.to(device).float()
            out = model(im1, im2).float()
            optimizer.zero_grad()
            loss = criterion(out, y)
            loss.backward()
            loss = loss.cpu().detach().numpy()
            losses.append(loss)
            optimizer.step()
            choose = np.argmax(out.cpu().detach().numpy(), axis=1)
            gt = y.cpu().detach().numpy()
            accuracy.append(len(np.where(gt == choose)[0]) / len(gt))

        losses_cumulative.append(np.mean(losses))
        accuracy_cumulative.append(np.mean(accuracy))
        print(np.mean(losses))
        print(np.mean(accuracy))
        losses = []
        accuracy = []
        # Do validation
        model.eval()
        for i, batch in enumerate(tqdm(test_dataloader)):
            x, y = batch
            im1, im2 = x
            y = y.to(device)
            im1 = im1.to(device).float()
            im2 = im2.to(device).float()
            out = model(im1, im2).float()
            #            out_class = Variable(out.argmax(1).float(), requires_grad=True)
            loss = criterion(out, y)
            loss = loss.cpu().detach().numpy()
            losses_v.append(loss)
            choose = np.argmax(out.cpu().detach().numpy(), axis=1)
            gt = y.cpu().detach().numpy()
            accuracy_v.append(len(np.where(gt == choose)[0]) / len(gt))
        losses_v_cumulative.append(np.mean(losses_v))
        accuracy_v_cumulative.append(np.mean(accuracy_v))
        print(np.mean(losses_v))
        print(np.mean(accuracy_v))
        losses_v = []
        accuracy_v = []
        if epoch % 5 == 0:
            torch.save(model.state_dict(), results_dir + "saved_model.pth")
            np.save(results_dir + "losses.npy", losses_cumulative)
            np.save(results_dir + "losses_v.npy", losses_v_cumulative)
            np.save(results_dir + "accuracy.npy", accuracy_cumulative)
            np.save(results_dir + "accuracy_v.npy", accuracy_v_cumulative)
    torch.save(model.state_dict(), results_dir + "saved_model.pth")
    np.save(results_dir + "losses.npy", losses_cumulative)
    np.save(results_dir + "losses_v.npy", losses_v_cumulative)
    np.save(results_dir + "accuracy.npy", accuracy_cumulative)
    np.save(results_dir + "accuracy_v.npy", accuracy_v_cumulative)
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
