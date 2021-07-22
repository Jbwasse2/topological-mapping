import argparse
import math
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

from data_getter import GibsonDataset
from model import Siamese
from rich.progress import track

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def soft_classification_loss(guess, truth):
    guess_class = guess.argmax(1)
    return torch.abs(guess_class - truth).sum().float()

def custom_MSELoss(guesses, truthes, similarityLabels, k=100):
    mse = nn.MSELoss()
    mask = similarityLabels.bool()
    guesses = guesses[mask,:]
    truthes = truthes[mask,:]
    #if there are no elements in the guesses/truthes will give nan
    #Should return loss of 0
    if guesses.numel() == 0:
        # set loss to 0, do this below to handle datastructures/gradients
        x = torch.rand(1,1)
        loss = mse(x,x)
    else:
        loss1 = mse(guesses[:,:3], truthes[:,:3])
        loss2 = mse(guesses[:,3], truthes[:,3]) * k
        loss = loss1 + loss2
    return loss.float()



def train(model, device, epochs=100):
    current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    results_dir = "../../data/results/sparsifier/" + current_time + "/"
    os.mkdir(results_dir)
    copyfile("./model.py", results_dir + "model.py")
    criterionSimilarity = nn.CrossEntropyLoss()
    criterionPose = custom_MSELoss 
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    BATCH_SIZE = 64
    seed = 0
    best_val_loss = np.inf
    train_dataset = GibsonDataset(
        "train",
        seed,
        samples=7000,
        max_distance=50,
        episodes=20,
        ignore_0=False,
        debug=False,
    )
    train_dataset.save_env_data(results_dir)
    train_dataloader = data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=10
    )
    test_dataset = GibsonDataset(
        "test",
        seed,
        samples=1000,
        max_distance=50,
        episodes=20,
        ignore_0=False,
        debug=False,
    )
    test_dataloader = data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=10,
    )
    train_acc = []
    losses_cumulative = []
    val_acc = []
    losses_v = []
    losses_v_cumulative = []
    pose_loss_cumulative = []
    pose_loss_v_cumulative = []
    accuracy_cumulative = []
    accuracy_v = []
    accuracy_v_cumulative = []
    for epoch in range(epochs):
        train_dataset.dataset = train_dataset.get_dataset(train_dataset.train_env)
        accuracy = []
        losses = []
        pose_loss = []
        print("epoch ", epoch)
        model.train()
        for i, batch in enumerate(tqdm(train_dataloader)):
            x, similarityGT, poseGT = batch
            im1, im2 = x
            hidden = model.init_hidden(similarityGT.shape[0], model.hidden_size, device)
            model.hidden = hidden
            similarityGT = similarityGT.to(device)
            poseGT = poseGT.to(device).float()
            im1 = im1.to(device).float()
            im2 = im2.to(device).float()
            similarity, pose = model(im1, im2)
            pose = pose.float()
            optimizer.zero_grad()
            loss1 = 10* criterionSimilarity(similarity, similarityGT).float()
            loss2 = criterionPose(pose, poseGT,similarityGT).float()
            loss = (loss1 + loss2).float()
            loss.backward()
            loss = loss.cpu().detach().numpy()
            losses.append(loss)
            optimizer.step()
            choose = np.argmax(similarity.cpu().detach().numpy(), axis=1)
            gt = similarityGT.cpu().detach().numpy()
            pose_loss.append(loss2.cpu().detach().item())
            accuracy.append(len(np.where(gt == choose)[0]) / len(gt))
        pose_loss_cumulative.append(np.mean(pose_loss))
        losses_cumulative.append(np.mean(losses))
        accuracy_cumulative.append(np.mean(accuracy))
        print("train")
        print("total loss")
        print(losses_cumulative[-1])
        print("similarity acc")
        print(accuracy_cumulative[-1])
        print("pose loss")
        print(pose_loss_cumulative[-1])
        losses = []
        accuracy = []
        pose_loss = []
        # Do validation
        model.eval()
        for i, batch in enumerate(tqdm(test_dataloader)):
            x, similarityGT, poseGT = batch
            im1, im2 = x
            hidden = model.init_hidden(similarityGT.shape[0], model.hidden_size, device)
            model.hidden = hidden
            similarityGT = similarityGT.to(device)
            poseGT = poseGT.to(device).float()
            im1 = im1.to(device).float()
            im2 = im2.to(device).float()
            similarity, pose = model(im1, im2)
            loss1 = 10 * criterionSimilarity(similarity, similarityGT).float()
            loss2 = criterionPose(pose, poseGT, similarityGT).float()
            loss = (loss1 + loss2).float()
            loss = loss.cpu().detach().numpy()
            losses_v.append(loss)
            choose = np.argmax(similarity.cpu().detach().numpy(), axis=1)
            gt = similarityGT.cpu().detach().numpy()
            pose_loss.append(loss2.cpu().detach().item())
            accuracy_v.append(len(np.where(gt == choose)[0]) / len(gt))
        pose_loss_v_cumulative.append(np.mean(pose_loss))
        losses_v_cumulative.append(np.mean(losses_v))
        accuracy_v_cumulative.append(np.mean(accuracy_v))
        print("Val")
        print("total loss")
        print(losses_v_cumulative[-1])
        print("similarity acc")
        print(accuracy_v_cumulative[-1])
        print("pose loss")
        print(pose_loss_v_cumulative[-1])
        losses_v = []
        accuracy_v = []
        if losses_v_cumulative[-1] < best_val_loss:
            best_val_loss = losses_v_cumulative[-1]
            print("New Best Model!")
            torch.save(model.state_dict(), results_dir + "saved_model.pth")
        if epoch % 5 == 0:
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
    model = Siamese(context=10).to(device)
    if args.network_location:
        model.load_state_dict(torch.load(args.network_location))
    model = train(model, device)
