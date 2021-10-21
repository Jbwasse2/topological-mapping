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
from model import SiameseDeepVO, Siamese
from rich.progress import track

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def soft_classification_loss(guess, truth):
    guess_class = guess.argmax(1)
    return torch.abs(guess_class - truth).sum().float()


def custom_MSELoss(guesses, truthes, similarityLabels, dist, k=1):
    mse = nn.MSELoss()
    mask = similarityLabels.bool()
    guesses = guesses[mask, :]
    truthes = truthes[mask, :]
    dist = dist[mask]
    T_GT = torch.norm(truthes[:, 0:3], dim=1)
    R_GT = truthes[:, 3]
    # if there are no elements in the guesses/truthes will give nan
    # Should return loss of 0
    if guesses.numel() == 0:
        # set loss to 0, do this below to handle datastructures/gradients
        x = torch.rand(1, 1)
        loss1 = mse(x, x)
        loss2 = mse(x, x)
    else:
        loss1 = 0
        for guess, gt, weight in zip(guesses[:, 0].flatten(), T_GT, dist):
            loss1 += mse(guess, gt)
        loss1 /= len(T_GT)
        loss1 *= k
        loss2 = 0
        for guess, gt, weight in zip(guesses[:, 1].flatten(), R_GT, dist):
            loss2 += mse(guess, gt)
        loss2 /= len(R_GT)
        loss2 *= k

    return loss1.float(), loss2.float()


def train(model, device, epochs=250):
    print("Switching dataset")
    current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    results_dir = "../../data/results/sparsifier/" + current_time + "/"
    os.mkdir(results_dir)
    copyfile("./model.py", results_dir + "model.py")
    criterionSimilarity = nn.CrossEntropyLoss()
    criterionPose = custom_MSELoss
    # optimizer = optim.Adam(model.parameters(), lr=0.0005)
    optimizer = optim.Adagrad(model.parameters(), lr=0.0005)
    BATCH_SIZE = 26
    DEBUG = False
    seed = 0
    best_val_loss = np.inf
    train_dataset = GibsonDataset(
        "train",
        seed,
        samples=100000 if not DEBUG else 202,
        max_distance=50,
        episodes=20,
        ignore_0=False,
        debug=DEBUG,
        give_distance=True,
    )
    train_dataset.save_env_data(results_dir)
    train_dataloader = data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=10 if not DEBUG else 0,
    )
    test_dataset = GibsonDataset(
        "test",
        seed,
        samples=10000 if not DEBUG else 202,
        max_distance=50,
        episodes=20,
        ignore_0=False,
        debug=DEBUG,
        give_distance=True,
    )
    test_dataloader = data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=10 if not DEBUG else 0,
    )
    train_acc = []
    losses_cumulative = []
    lossesT_cumulative = []
    lossesR_cumulative = []
    lossesS_cumulative = []
    lossesT_v_cumulative = []
    lossesR_v_cumulative = []
    lossesS_v_cumulative = []
    val_acc = []
    losses_v = []
    losses_v_cumulative = []
    pose_loss_cumulative = []
    pose_loss_v_cumulative = []
    accuracy_cumulative = []
    accuracy_v = []
    accuracy_v_cumulative = []
    for epoch in range(epochs):
        if epoch >= 0:
            train_dataset.dataset = train_dataset.get_dataset_balanced(
                train_dataset.train_env
            )
        accuracy = []
        losses = []
        lossesT = []
        lossesR = []
        lossesS = []
        print("epoch ", epoch)
        model.train()
        for i, batch in enumerate(tqdm(train_dataloader)):
            x, similarityGT, dist, poseGT = batch
            im1, im2 = x
            hidden = model.init_hidden(similarityGT.shape[0], model.hidden_size, device)
            model.hidden = hidden
            similarityGT = similarityGT.to(device)
            poseGT = poseGT.to(device).float()
            im1 = im1.to(device).float()
            im2 = im2.to(device).float()
            pose, similarity = model(im1, im2, poseGT)
            pose = pose.float()
            optimizer.zero_grad()
            lossS = 1 * criterionSimilarity(similarity, similarityGT).float()
            lossT, lossR = criterionPose(pose, poseGT, similarityGT, dist)
            #            loss = (50 * lossT + 50 * 5 * lossR + lossS).float()
            loss = (lossT + 5 * lossR + lossS).float()
            try:
                loss.backward()
                loss = loss.cpu().detach().numpy().item()
                lossT = lossT.cpu().detach().numpy().item()
                lossR = lossR.cpu().detach().numpy().item()
                lossS = lossS.cpu().detach().numpy().item()
                losses.append(loss)
                lossesT.append(lossT)
                lossesR.append(lossR)
                lossesS.append(lossS)
                optimizer.step()
            except Exception as e:
                print(e)
            choose = np.argmax(similarity.cpu().detach().numpy(), axis=1)
            gt = similarityGT.cpu().detach().numpy()
            accuracy.append(len(np.where(gt == choose)[0]) / len(gt))
        losses_cumulative.extend(losses)
        lossesT_cumulative.extend(lossesT)
        lossesR_cumulative.extend(lossesR)
        lossesS_cumulative.extend(lossesS)
        accuracy_cumulative.extend(accuracy)
        print("train")
        print("T loss")
        print(lossesT_cumulative[-1])
        print("R loss")
        print(lossesR_cumulative[-1])
        print("S loss")
        print(lossesS_cumulative[-1])
        print("similarity acc")
        print(accuracy_cumulative[-1])
        lossesT = []
        lossesR = []
        lossesS = []
        accuracy = []
        pose_loss = []
        losses_v = []
        # Do validation
        model.eval()
        for i, batch in enumerate(tqdm(test_dataloader)):
            x, similarityGT, dist, poseGT = batch
            im1, im2 = x
            hidden = model.init_hidden(similarityGT.shape[0], model.hidden_size, device)
            model.hidden = hidden
            similarityGT = similarityGT.to(device)
            poseGT = poseGT.to(device).float()
            im1 = im1.to(device).float()
            im2 = im2.to(device).float()
            pose, similarity = model(im1, im2, poseGT)
            lossS = 1 * criterionSimilarity(similarity, similarityGT).float()
            lossT, lossR = criterionPose(pose, poseGT, similarityGT, dist)
            loss = (lossT + 5 * lossR + lossS).float()
            loss = loss.cpu().detach().numpy()
            losses_v.append(loss)
            lossT = lossT.cpu().detach().numpy()
            lossR = lossR.cpu().detach().numpy()
            lossS = lossS.cpu().detach().numpy()
            lossesT.append(lossT)
            lossesR.append(lossR)
            lossesS.append(lossS)
            choose = np.argmax(similarity.cpu().detach().numpy(), axis=1)
            gt = similarityGT.cpu().detach().numpy()
            accuracy_v.append(len(np.where(gt == choose)[0]) / len(gt))
        losses_v_cumulative.extend(losses_v)
        lossesT_v_cumulative.extend(lossesT)
        lossesR_v_cumulative.extend(lossesR)
        lossesS_v_cumulative.extend(lossesS)
        accuracy_v_cumulative.extend(accuracy_v)
        print("Val")
        print("T loss")
        print(lossesT_v_cumulative[-1])
        print("R loss")
        print(lossesR_v_cumulative[-1])
        print("S loss")
        print(lossesS_v_cumulative[-1])
        print("similarity acc")
        print(accuracy_v_cumulative[-1])
        losses_v = []
        accuracy_v = []
        if losses_v_cumulative[-1] < best_val_loss:
            best_val_loss = losses_v_cumulative[-1]
            print("New Best Model!")
            torch.save(model.state_dict(), results_dir + "saved_model.pth")
        if epoch % 1 == 0:
            np.save(results_dir + "losses.npy", losses_cumulative)
            np.save(results_dir + "losses_v.npy", losses_v_cumulative)
            np.save(results_dir + "accuracy.npy", accuracy_cumulative)
            np.save(results_dir + "accuracy_v.npy", accuracy_v_cumulative)
            np.save(results_dir + "lossesR.npy", lossesR_cumulative)
            np.save(results_dir + "lossesR_v.npy", lossesR_v_cumulative)
            np.save(results_dir + "lossesT.npy", lossesT_cumulative)
            np.save(results_dir + "lossesT_v.npy", lossesT_v_cumulative)
            np.save(results_dir + "lossesS.npy", lossesS_cumulative)
            np.save(results_dir + "lossesS_v.npy", lossesS_v_cumulative)
            print(results_dir)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--network_location", help="No", required=False)
    args = parser.parse_args()
    device = torch.device("cuda")
    # model = Siamese().to(device)
    model = SiameseDeepVO().to(device)
    model.device = device
    pretrained = "./flownets_EPE1.951.pth.tar"
    pretrained_w = torch.load(pretrained)
    print("Load FlowNet pretrained model")
    # Use only conv-layer-part of FlowNet as CNN for DeepVO
    model_dict = model.state_dict()
    update_dict = {
        k: v for k, v in pretrained_w["state_dict"].items() if k in model_dict
    }
    model_dict.update(update_dict)
    model.load_state_dict(model_dict)
    if args.network_location:
        model.load_state_dict(torch.load(args.network_location))
        print("Loaded in model!")
    model = train(model, device)
