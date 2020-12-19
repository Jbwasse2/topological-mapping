import gzip
import itertools
import os

import cv2
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pudb
import seaborn as sns
import torch
from tqdm import tqdm

from model.model import Siamese
from test_data import GibsonMapDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sns.set(style="darkgrid", font_scale=1.5)
matplotlib.rcParams["font.family"] = "Helvetica"
font = {"weight": "bold"}

matplotlib.rc("font", **font)


def get_dict(fname):
    f = gzip.open(
        "../../data/datasets/pointnav/gibson/v2/train_large/content/"
        + fname
        + ".json.gz"
    )
    content = f.read()
    content = content.decode()
    content = content.replace("null", "None")
    content = eval(content)
    return content["episodes"]


# Takes images of trajectory and sparsifies them
def sparsify_trajectories(model, trajs, trajs_visual, device):
    traj_ind = []
    for traj in tqdm(trajs):
        image_indices = [0]
        skip_index = 0
        for i in range(len(traj)):
            if i < skip_index:
                continue
            image1 = traj[i].unsqueeze(0).float().to(device)
            j = 0
            while 1:
                if i + j + 1 >= len(traj):
                    break
                image2 = traj[i + 1 + j].unsqueeze(0).float().to(device)
                j += 1
                results = np.argmax(model(image1, image2).cpu().detach().numpy())
                if results == 0:
                    skip_index = i + j
                    image_indices.append(skip_index)
                    break
        traj_ind.append(image_indices)
    ret = []
    for traj, indices in tqdm(zip(trajs_visual, traj_ind)):
        ret.append([traj[i] for i in indices])
    return ret, traj_ind


# Gets all of the trajectories in the environment
def get_trajectory_env(data, env):
    data.set_env(env)
    counter = 0
    trajs = []
    for i in tqdm(range(data.number_of_trajectories)):
        done = False
        traj = []
        while not done:
            im, done = data[counter]
            traj.append(im)
            counter += 1
        trajs.append(traj)
    return trajs


def visualize_traj(traj, name):
    # Create video of traj
    results_dir = "../../data/results/map/"
    video_name = results_dir + name + ".mkv"
    height, width, _ = traj[0].shape
    video = cv2.VideoWriter(video_name, 0, 10, (width, height))
    for frame in traj:
        video.write(frame)
    cv2.destroyAllWindows()
    video.release()
    return 1


def visualize_traj_ind(traj_ind):
    fig, ax = plt.subplots()
    distance = []
    results_dir = "../../data/results/map/"
    for traj in traj_ind:
        for i, j in zip(traj, traj[1:]):
            distance.append(j - i)
    N, bins, patches = ax.hist(
        distance, edgecolor="white", bins=np.arange(max(distance) - 0.0)
    )
    for i in range(0, 5):
        patches[i].set_facecolor("r")
    plt.xticks(np.arange(0, max(distance), 1.0))
    plt.savefig(results_dir + "traj_ind.jpg", bbox_inches="tight")


def main():
    VISUALIZE = True
    device = torch.device("cuda:0")
    model = Siamese().to(device)
    model.load_state_dict(torch.load("./model/saved_model.pth"))
    model.eval()
    test_envs = np.load("./model/test_env.npy")
    data = GibsonMapDataset(test_envs)
    trajs = get_trajectory_env(data, test_envs[0])
    data_visual = GibsonMapDataset(test_envs, transform=False)
    trajs_visual = get_trajectory_env(data_visual, test_envs[0])
    traj_new, traj_ind = sparsify_trajectories(model, trajs, trajs_visual, device)
    if VISUALIZE:
        visualize_traj_ind(traj_ind)
        visualize_traj(trajs_visual[0], "old_traj")
        visualize_traj(traj_new[0], "new_traj")


if __name__ == "__main__":

    main()
