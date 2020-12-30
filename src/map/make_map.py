import gzip
import pathlib
import shutil
import habitat
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

from habitat.datasets.utils import get_action_shortest_path

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
matplotlib.rcParams["font.family"] = "Helvetica"
font = {"weight": "bold"}

matplotlib.rc("font", **font)


def create_sim(scene):
    cfg = habitat.get_config()
    cfg.defrost()
    cfg.SIMULATOR.SCENE = "../../data/scene_datasets/gibson/" + scene + ".glb"
    cfg.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR"]
    cfg.freeze()
    sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)
    return sim


# Returns the length of the shortest path between two edges
def actual_edge_len(edge, d, sim):
    # Get starting position and rotation
    node1 = edge[0]
    pose1 = d[node1[0]]["shortest_paths"][0][node1[1]]
    position1 = pose1["position"]
    rotation1 = pose1["rotation"]
    # Get ending position
    node2 = edge[1]
    pose2 = d[node2[0]]["shortest_paths"][0][node2[1]]
    position2 = pose2["position"]
    rotation2 = pose2["rotation"]
    # Get actual distance from simulator, use same settings as the one used in data collection
    if position1 == position2:
        return 0
    shortest_path_success_distance = 0.2
    shortest_path_max_steps = 500
    shortest_path = get_action_shortest_path(
        sim,
        source_position=position1,
        source_rotation=rotation1,
        goal_position=position2,
        success_distance=shortest_path_success_distance,
        max_episode_steps=shortest_path_max_steps,
    )
    return len(shortest_path)


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
def sparsify_trajectories(model, trajs, trajs_visual, device, sparsity=4):
    #        trajs=np.load("trajs.npy", allow_pickle=True)
    pu.db
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
                if results > sparsity + 10:
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
    #    for i in tqdm(range(data.number_of_trajectories)):
    for i in range(3):
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


# Visualize distances between trajectories, should mostly be close to sparisity threshold set.
def visualize_traj_ind(traj_ind):
    sns.set(style="darkgrid", font_scale=1.5)
    distance = []
    for traj in traj_ind:
        for i, j in zip(traj, traj[1:]):
            distance.append(j - i)
    fig, ax = plt.subplots()
    results_dir = "../../data/results/map/"
    N, bins, patches = ax.hist(
        distance, edgecolor="white", bins=np.arange(max(distance) - 0.0)
    )
    for i in range(0, 5):
        patches[i].set_facecolor("r")
    plt.xticks(np.arange(0, max(distance), 1.0))
    plt.savefig(results_dir + "traj_ind.jpg", bbox_inches="tight")


# Takes node label (0,0) and returns corresponding image as 640x480, useful for visulaizations.
def get_node_image(node, scene):
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


# Takes a pair of nodes and creates a visualzation.
# Also takes a string for name
def visualize_edge(node1, node2, scene, out_file_name):
    sns.set(style="dark", font_scale=1.0)
    im1 = get_node_image(node1, scene)
    im2 = get_node_image(node2, scene)
    f, axes = plt.subplots(1, 2)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].imshow(im1)
    axes[1].imshow(im2)
    plt.savefig(out_file_name, bbox_inches="tight")


# Takes a list of trajectories and adds them to the graph
# This does not make connecitons between the trajectories, just adds them.
# Does not add node in trajectory if node is too similar to another node in graph
def add_trajs_to_graph(G, trajectories, traj_ind, model, device, similarity):
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
                if results <= similarity + 10:
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
def connect_graph_trajectories(
    G, trajectories, traj_ind, model, device, similarity, sim=None, d=None, scene=None
):
    img_dir = "../../data/results/map/edges/"
    shutil.rmtree(img_dir)
    pathlib.Path(img_dir).mkdir(parents=True, exist_ok=False)
    counter = 0
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
            if result_close <= similarity + 10:
                G.add_edge(node_i, node_j)
                # Check for wormholes
                if sim != None and d != None:
                    true_length = actual_edge_len((node_i, node_j), d, sim)
                    # Perform visualizations
                    if scene != None:
                        visualize_edge(
                            node_i,
                            node_j,
                            scene,
                            img_dir
                            + "edge_"
                            + str(counter).zfill(4)
                            + "_pred_"
                            + str(result_close)
                            + "_true_"
                            + str(true_length)
                            + ".png",
                        )
                        counter += 1
    return G


def create_topological_map(
    trajectories, traj_ind, model, device, similarity=2, sim=None, d=None, scene=None
):
    # Put all trajectories into graph
    G = nx.DiGraph()
    G = add_trajs_to_graph(G, trajectories, traj_ind, model, device, similarity)
    G = connect_graph_trajectories(
        G, trajectories, traj_ind, model, device, similarity, sim=sim, d=d, scene=scene
    )
    return G


def main():
    VISUALIZE = True
    CREATE_TRAJECTORIES = True
    device = torch.device("cuda:0")
    model = Siamese().to(device)
    model.load_state_dict(torch.load("./model/saved_model.pth"))
    model.eval()
    test_envs = np.load("./model/test_env.npy")
    ENV = test_envs[0]
    if CREATE_TRAJECTORIES == True:
        data = GibsonMapDataset(test_envs)
        # trajs is the trajectory of the 224x224 dataset, not sparsified
        trajs = get_trajectory_env(data, ENV)
        # traj_new is sparsified trajectory with the traj_visual dataset
        # traj_ind says which indices where used
        traj_new, traj_ind = sparsify_trajectories(
            model, trajs, trajs, device, sparsity=4
        )
        np.save("traj_new.npy", traj_new)
        np.save("traj_ind.npy", traj_ind)
        np.save("trajs.npy", trajs)
    else:
        traj_new = np.load("traj_new.npy", allow_pickle=True)
        traj_ind = np.load("traj_ind.npy", allow_pickle=True)
    if VISUALIZE:
        visualize_traj_ind(traj_ind)

    sim = create_sim(ENV)
    G = create_topological_map(
        traj_new,
        traj_ind,
        model,
        device,
        similarity=2,
        sim=sim,
        d=get_dict(ENV),
        scene=ENV,
    )

    print(ENV)
    nx.write_gpickle(G, "../../data/map/map_" + str(ENV) + ".gpickle")


if __name__ == "__main__":
    main()
