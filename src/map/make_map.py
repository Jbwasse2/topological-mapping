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


# Takes node label (0,0) and returns corresponding image as 640x480, useful for visulaizations.
def get_node_image(node):
    image_location = (
        "../../data/datasets/pointnav/gibson/v2/train_large/images/"
        + scene
        + "/"
        + "episode"
        + str(node[0])
        + "_"
        + str(node[1]).zfill(5)
        + ".jpg"
    )
    return plt.imread(image_location)


def get_true_label(trajectory_count, frame_count, traj_ind):
    return traj_ind[trajectory_count][frame_count]


# Takes a list of trajectories and adds them to the graph
# This does not make connecitons between the trajectories, just adds them.
def add_trajs_to_graph(G, trajectories, traj_ind, model, device):
    for trajectory_count, trajectory in tqdm(enumerate(trajectories)):
        tail_node = None
        for frame_count, frame in enumerate(trajectory):
            # we want to keep track of what the actual frame is from in the trajectory
            true_label = get_true_label(trajectory_count, frame_count, traj_ind)
            current_node = (trajectory_count, true_label)
            image1 = frame.unsqueeze(0).float().to(device)
            for node in list(G.nodes):
                # Skip previous tail node from being too close to current node, otherwise you may just be sparsifying your graph extra.
                if node == tail_node:
                    continue
                # Get the index in the trajectory where node is (sparse label)
                node_traj_ind = traj_ind[node[0]]
                node_ind = node_traj_ind.index(node[1])
                # Get original node image
                node_image = trajectories[node[0]][node_ind]
                image2 = node_image.unsqueeze(0).float().to(device)
                results = np.argmax(model(image1, image2).cpu().detach().numpy())
                if results == 1:
                    current_node = node
                    break
            # Check to see if there is already a similar enough edge in the graph
            G.add_node(current_node)
            if tail_node is not None:
                G.add_edge(tail_node, current_node)
            else:
                pass
            tail_node = current_node
    return G


# Note: add_traj_to_graph needs to be called first.
# Takes the trajectories and connects them together by using the sparsifier
def connect_graph_trajectories(G, trajectories, traj_ind, model, device):
    for node_i in tqdm(list(G.nodes)):
        node_traj_ind_i = traj_ind[node_i[0]]
        node_ind_i = node_traj_ind_i.index(node_i[1])
        # Get original node image
        node_image_i = (
            trajectories[node_i[0]][node_ind_i].unsqueeze(0).float().to(device)
        )
        for node_j in list(G.nodes):
            if node_i == node_j:
                continue
            node_traj_ind_j = traj_ind[node_j[0]]
            node_ind_j = node_traj_ind_j.index(node_j[1])
            # Get original node image
            node_image_j = (
                trajectories[node_j[0]][node_ind_j].unsqueeze(0).float().to(device)
            )
            result = model(node_image_i, node_image_j).cpu().detach().numpy()
            result_close = np.argmax(result)
            if result_close == 1:
                G.add_edge(node_i, node_j)
    return G


def create_topological_map(trajectories, traj_ind, model, device):
    # Put all trajectories into graph
    G = nx.DiGraph()
    G = add_trajs_to_graph(G, trajectories, traj_ind, model, device)
    G = connect_graph_trajectories(G, trajectories, traj_ind, model, device)
    return G


def main():
    VISUALIZE = True
    CREATE_TRAJECTORIES = False
    device = torch.device("cuda:0")
    model = Siamese().to(device)
    model.load_state_dict(torch.load("./model/saved_model.pth"))
    model.eval()
    if CREATE_TRAJECTORIES == True:
        test_envs = np.load("./model/test_env.npy")
        data = GibsonMapDataset(test_envs)
        # trajs is the trajectory of the 224x224 dataset, not sparsified
        trajs = get_trajectory_env(data, test_envs[0])
        # traj_new is sparsified trajectory with the traj_visual dataset
        # traj_ind says which indices where used
        traj_new, traj_ind = sparsify_trajectories(model, trajs, trajs, device)
        np.save("traj_new.npy", traj_new)
        np.save("traj_ind.npy", traj_ind)
        np.save("trajs.npy", trajs)
    else:
        traj_new = np.load("traj_new.npy", allow_pickle=True)
        traj_ind = np.load("traj_ind.npy", allow_pickle=True)
    #        trajs_visual=np.load("trajs_visual.npy", allow_pickle=True)
    #        trajs=np.load("trajs.npy", allow_pickle=True)
    G = create_topological_map(traj_new, traj_ind, model, device)
    nx.write_gpickle(G, "../../data/map/map2.gpickle")

    if VISUALIZE:
        # trajs_visual is the 640x480 version of teh dataset, used for visualization mostly
        trajs_visual = get_trajectory_env(data_visual, test_envs[0])
        data_visual = GibsonMapDataset(test_envs, transform=False)
        traj_new, traj_ind = sparsify_trajectories(model, trajs, trajs_visual, device)
        visualize_traj_ind(traj_ind)
        visualize_traj(trajs_visual[0], "old_traj")
        visualize_traj(traj_new[0], "new_traj")


if __name__ == "__main__":
    main()
