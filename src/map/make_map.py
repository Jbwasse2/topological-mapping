import gzip
import random
import pathlib
import shutil
import habitat
import itertools
import os

import quaternion
from habitat.utils.geometry_utils import (
    quaternion_to_list,
    angle_between_quaternions,
    quaternion_from_coeff,
)
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
    if position1 == position2 and rotation1 == rotation2:
        return 0
    shortest_path_success_distance = 0.2
    shortest_path_max_steps = 500
    shortest_path, final_state_rotation = get_action_shortest_path(
        sim,
        source_position=position1,
        source_rotation=rotation1,
        goal_position=position2,
        success_distance=shortest_path_success_distance,
        max_episode_steps=shortest_path_max_steps,
    )
    if not isinstance(rotation2, quaternion.quaternion):
        rotation2 = quaternion_from_coeff(rotation2)
    angle = angle_between_quaternions(final_state_rotation, rotation2)
    actuation_rotation = np.deg2rad(
        sim.config.agents[0].action_space[2].actuation.amount
    )
    return len(shortest_path) + int(angle / actuation_rotation)


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
# Assumes we have perfect knowledge of distance, but this doesn't matter for localization problem.
def sparsify_trajectories(model, trajs, trajs_visual, device, d, sim, sparsity=4):
    traj_ind = []
    for traj in tqdm(trajs):
        image_indices = []
        for i in range(len(traj)):
            if i % sparsity == 0:
                image_indices.append(i)
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
def add_trajs_to_graph(
    G, trajectories, traj_ind, model, device, similarity, episodes, d, sim
):
    for trajectory_count, (trajectory_count_episode, trajectory) in tqdm(
        enumerate(zip(episodes, trajectories))
    ):
        tail_node = None
        for frame_count, frame in enumerate(trajectory):
            # we want to keep track of what the actual frame is from in the trajectory
            true_label = get_true_label(trajectory_count, frame_count, traj_ind)
            current_node = (trajectory_count_episode, true_label)
            for node in list(G.nodes):
                # Skip previous tail node from being too close to current node, otherwise you may just be sparsifying your graph extra.
                if node == tail_node:
                    continue
                edge = (current_node, node)
                results = actual_edge_len(edge, d, sim)
                if results <= similarity:
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
    G, trajectories, traj_ind, model, device, similarity, episodes, sim=None, d=None
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
            # result = model(node_image_i, node_image_j).cpu().detach().numpy()
            results = actual_edge_len(edge, d, sim)
            result_close = np.argmax(result)
            if result_close <= similarity:
                G.add_edge(node_i, node_j)
    return G


def create_topological_map(
    trajectories,
    traj_ind,
    model,
    device,
    episodes,
    similarity=2,
    sim=None,
    d=None,
    scene=None,
):
    # Put all trajectories into graph
    G = nx.DiGraph()
    G = add_trajs_to_graph(
        G, trajectories, traj_ind, model, device, similarity, episodes, sim=sim, d=d
    )
    return G


def build_graph(hold_out_percent=0.10):
    VISUALIZE = False
    CREATE_TRAJECTORIES = True
    device = torch.device("cuda:0")
    model = Siamese().to(device)
    model.load_state_dict(torch.load("./model/saved_model.pth"))
    model.eval()
    test_envs = np.load("./model/test_env.npy")
    ENV = test_envs[0]
    d = get_dict(ENV)
    sim = create_sim(ENV)

    if CREATE_TRAJECTORIES == True:
        data = GibsonMapDataset(test_envs)
        # trajs is the trajectory of the 224x224 dataset, not sparsified
        trajs = get_trajectory_env(data, ENV)
        # traj_new is sparsified trajectory with the traj_visual dataset
        # traj_ind says which indices where used
        traj_new, traj_ind = sparsify_trajectories(
            model, trajs, trajs, device, d, sim, sparsity=4
        )
        np.save("traj_new.npy", traj_new)
        np.save("traj_ind.npy", traj_ind)
        np.save("trajs.npy", trajs)
    else:
        traj_new = np.load("traj_new.npy", allow_pickle=True)
        traj_ind = np.load("traj_ind.npy", allow_pickle=True)
    if VISUALIZE:
        visualize_traj_ind(traj_ind)
    # some trajectories will be used to build the map, some will be used to evaluate the failure of imagegoal
    traj_held_out = int(len(traj_new) * hold_out_percent)
    eval_trajs = random.choices(list(range(len(traj_ind))), k=traj_held_out)
    map_trajs = list(set(range(len(traj_ind))) - set(eval_trajs))

    traj_new_map = [traj_new[i] for i in map_trajs]
    traj_ind_map = [traj_ind[i] for i in map_trajs]
    traj_new_eval = [traj_new[i] for i in eval_trajs]
    traj_ind_eval = [traj_ind[i] for i in eval_trajs]

    assert len(traj_new_map) == len(traj_ind_map)
    assert len(traj_new_eval) == len(traj_ind_eval)
    np.save("traj_new_eval.npy", traj_new_eval)
    np.save("traj_ind_eval.npy", traj_ind_eval)
    G = create_topological_map(
        traj_new_map,
        traj_ind_map,
        model,
        device,
        episodes=map_trajs,
        similarity=4,
        sim=sim,
        d=d,
        scene=ENV,
    )

    print(ENV)
    nx.write_gpickle(G, "../../data/map/map_" + str(ENV) + ".gpickle")
    return G, traj_new_eval, traj_ind_eval


def find_wormholes(G):
    pass


if __name__ == "__main__":
    G = build_graph(0.10)
    wormholes = find_wormholes(G)
