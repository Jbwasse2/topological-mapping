import math
import pickle
from glob import glob
import os
import time

import matplotlib
import matplotlib.pyplot as plt
from torchvision import transforms as transforms
import cv2
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import gc
from tqdm import tqdm

import habitat
from habitat.utils.geometry_utils import angle_between_quaternions
from make_map import (create_sim, get_dict, get_node_image, get_trajectory_env,
                      get_true_label, se3_to_habitat, sparsify_trajectories)
from test_data import GibsonMapDataset
from worm_model.model import Siamese

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
matplotlib.rcParams["font.family"] = "Helvetica"
font = {"weight": "bold"}

def get_node_image_sequence(node, scene, max_lengths, transform=False, context=10):
    ret = []
    trnsform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    #Probably wanna cache this later...
    if node[0] not in max_lengths:
        max_length = len(glob(
                 (
                "../../data/datasets/pointnav/gibson/v4/train_large/images/"
                + scene
                + "/"
                + "episodeRGB"
                + str(node[0])
                + "_*.jpg"
            )))
        max_lengths[node[0]] = max_length
    else:
        max_length = max_lengths[node[0]]
    for i in range(node[1] - context, node[1]):
        i = max(i, 0)
        i = min(i, max_length)
        image_location = (
            "../../data/datasets/pointnav/gibson/v4/train_large/images/"
            + scene
            + "/"
            + "episodeRGB"
            + str(node[0])
            + "_"
            + str(i).zfill(5)
            + ".jpg"
        )
        image = plt.imread(image_location)
        image = cv2.resize(image, (224, 224)) / 255
        if transform:
            ret.append(trnsform(image))
        else:
            ret.append(image)
    return torch.stack(ret), max_lengths


def estimate_edge_len_wormhole(model, edges, sim, device, max_lengths, buffer_size):
    # Get images for edges
    if edges == []:
        return []
    images1 = torch.zeros(buffer_size, 10, 3, 224, 224)
    images2 = torch.zeros(buffer_size, 10, 3, 224, 224)
    for count, edge in enumerate(edges):
        scene = os.path.basename(sim.config.sim_cfg.scene.id).split(".")[0]
        images1[count], max_lengths = get_node_image_sequence(edge[0], scene, transform=True, max_lengths=max_lengths)
        images2[count], max_lengths = get_node_image_sequence(edge[1], scene, transform=True, max_lengths=max_lengths)
    images1 = (images1).float().to(device)
    images2 = (images2).float().to(device)
    if len(images1.shape) == 4:
        images1 = images1.unsqueeze(0)
        images2 = images2.unsqueeze(0)
    hidden = model.init_hidden(images1.shape[0], model.hidden_size, device)
    model.hidden = hidden
    similarity, pose = model(images1, images2)
    similarity = similarity[0:len(edges)]
    pose = pose[0:len(edges)]
    similarity = F.softmax(similarity)
    return similarity[:, 1].detach().cpu().numpy(), pose.detach().cpu().numpy(), max_lengths


def estimate_edge_len_pose(d, edges):
    ret = []
    closest_node = (np.inf, None)
    for counter, edge in enumerate(edges):
        node1 = edge[0]
        pose1 = d[node1[0]][node1[1]]
        if pose1 == None:
            continue
        position1 = pose1["position"]
        rotation1 = pose1["rotation"]
        node2 = edge[1]
        pose2 = d[node2[0]][node2[1]]
        if pose2 == None:
            continue
        position2 = pose2["position"]
        rotation2 = pose2["rotation"]
        POSITION_GRANULARITY = 0.25 / 5  # meters
        ANGLE_GRANULARITY = math.radians(10 / 5)
        rot_diff = angle_between_quaternions(rotation1, rotation2)
        distance_diff = np.linalg.norm(position1 - position2)
        distance = int(distance_diff / POSITION_GRANULARITY) + int(
            rot_diff / ANGLE_GRANULARITY
        )
        if distance < 15:
            ret.append(1)
        else:
            ret.append(0)
        if distance < closest_node[0]:
            closest_node = (distance, edge)
    return closest_node, ret


# If a node in the graph is already similiar to the node being proposed, use the ndoe in graph isntead. This will just return 1 node that is close
def find_close_node(model, edges, sim, device, similarity, map_type, max_lengths, buffer_size):
    pu.db
    return_node = None
    if map_type == "topological":
        results, pose, max_lengths = estimate_edge_len_wormhole(model, edges, sim, device, max_lengths, buffer_size)
        for counter, result in enumerate(results):
            if result >= similarity:
                return_node = counter
                break
    elif map_type == "pose":
        results = estimate_edge_len_pose(d, edges)[0]
        if results[0] < 15:
            return_node = edges.index(results[1])
    return return_node, max_lengths


# This will return all nodes that are close
def find_close_nodes(model, edges, sim, device, similarity, map_type, max_lengths, buffer_size):
    if map_type == "topological":
        results, pose, max_lengths = estimate_edge_len_wormhole(model, edges, sim, device, max_lengths, buffer_size)
        return_edges = []
        for counter, result in enumerate(results):
            if result >= similarity:
                return_edges.append(counter)
        return return_edges, max_lengths
    elif map_type == "pose":
        results = estimate_edge_len_pose(d, edges)[1]
        return [i for i, x in enumerate(results) if x == 1]


def add_trajs_to_graph_wormhole(
    G,
    trajectories,
    traj_ind,
    model,
    device,
    similarity,
    episodes,
    sim,
    buffer_size=200,
    map_type="topological",
):
    max_lengths = {}
    tail_node = None
    for trajectory_count, (trajectory_count_episode, trajectory) in tqdm(
        enumerate(zip(episodes, trajectories))
    ):
        for frame_count, frame in tqdm(enumerate(trajectory)):
            # we want to keep track of what the actual frame is from in the trajectory
            true_label = get_true_label(trajectory_count, frame_count, traj_ind)
            current_node = (trajectory_count_episode, true_label)
            nodes = list(G.nodes)
            edges = []
            if frame_count % 30 == 0:
                if tail_node != None:
                    edges.append((tail_node, current_node) )
                    prob_similar, pose, max_lengths = estimate_edge_len_wormhole(model, edges, sim, device, max_lengths, buffer_size=len(edges))
                    pose = pose[0]
                    tail_node_global = tail_node[3]
                    global_pose = (pose[0] + tail_node_global[0], pose[1] + tail_node_global[1], pose[2] + tail_node_global[2], (pose[3] + tail_node_global[3] % np.pi))
                    current_node = (current_node[0], current_node[1], tuple(pose), global_pose)
                    G.add_node(current_node)
                    G.add_edge(tail_node, current_node)
                else:
                    current_node = (current_node[0], current_node[1], (0.0,0.0,0.0,0.0), (0.0,0.0,0.0,0.0))
                    G.add_node(current_node)
                tail_node = current_node
    return G


def create_topological_map_wormhole(
    trajectories,
    traj_ind,
    model,
    device,
    episodes,
    similarityNodes,
    similarityEdges,
    sim=None,
    d=None,
    scene=None,
    map_type="topological",
):
    # Put all trajectories into graph
    G = nx.DiGraph()
    d = get_dict(scene)
    G = add_trajs_to_graph_wormhole(
        G,
        trajectories,
        traj_ind,
        model,
        device,
        similarityNodes,
        episodes,
        sim=sim,
        map_type=map_type,
    )
    G = connect_graph_trajectories_wormholes(
        G,
        model,
        device,
        similarityEdges,
        sim=sim,
        map_type=map_type,
    )

    return G


def connect_graph_trajectories_wormholes(
    G, model, device, similarity, sim=None, buffer_size=200, map_type="topological", closeness = 5.0
):
    counter = 0
    nodes = list(G.nodes)
    max_lengths = {}
    for node_i in tqdm(nodes):
        edges = []
        for i, node_j in enumerate(list(G.nodes)):
            #Only add nodes to edges if the predicted pose is close...
            distance = np.linalg.norm(np.array(node_i[3]) - np.array(node_j[3]))
            if distance >= closeness or node_i == node_j:
                continue
            edge = (node_i, node_j)
            edges.append(edge)
            if (
                len(edges) >= buffer_size or i == len(nodes) - 1
            ):  # -2 because range is over n-1 and then one node should be tail node
                if len(edges) >= buffer_size:
                    pu.db
                print(len(edges))
                try:
                    pu.db
                    results, max_lengths = find_close_nodes(
                        model, edges, sim, device, similarity, map_type=map_type, max_lengths=max_lengths, buffer_size=buffer_size
                    )
                except Exception as e:
                    print(len(edges))
                    print(e)
                    pu.db
                for result in results:
                    edge = edges[result]
                    if edge[0] == edge[1]:
                        continue
                    G.add_edge(edge[0], edge[1])
                edges = []
    return G


def main(env, similarityNodes=0.99, similarityEdges=0.80, map_type="topological"):
    device = torch.device("cuda:0")
    test_envs = np.load("./worm_model/test_env.npy")
    sim = create_sim(env)
    sparsifier_model = Siamese().to(device)
    sparsifier_model.load_state_dict(torch.load("./worm_model/saved_model.pth"))
    sparsifier_model.eval()
    data = GibsonMapDataset(test_envs)
    # trajs is the trajectory of the 224x224 dataset, not sparsified
    trajs = get_trajectory_env(data, env, number_of_trajectories=20)
    # traj_new is sparsified trajectory with the traj_visual dataset
    # traj_ind says which indices where used
    traj_new, traj_ind = sparsify_trajectories(
        trajs,
        device,
        sim,
        sparsity=1,
    )
    episodes = list(range(len(traj_new)))
    G = create_topological_map_wormhole(
        traj_new,
        traj_ind,
        sparsifier_model,
        episodes=episodes,
        similarityNodes=similarityNodes,
        similarityEdges=similarityEdges,
        sim=sim,
        scene=env,
        device=device,
        map_type=map_type,
    )
    if map_type == "topological":
        nx.write_gpickle(
            G,
            "../../data/map/mapWorm20NewArch_" + str(env) + str(similarityEdges) + ".gpickle",
        )
    elif map_type == "pose":
        nx.write_gpickle(
            G,
            "../../data/map/mapWormPose50_"
            + str(env)
            + str(similarityEdges)
            + ".gpickle",
        )
    return G


def find_wormholes(G, d, wormhole_distance=5.0):
    wormholes = []
    for edge in list(G.edges()):
        node1, node2 = edge[0], edge[1]
        if node1 == (1,60) and node2 == (16,45):
            pu.db
        position1 = d[node1[0]]["shortest_paths"][0][0][node1[1]]["position"]
        position2 = d[node2[0]]["shortest_paths"][0][0][node2[1]]["position"]
        distance = np.linalg.norm(np.array(position1) - np.array(position2))
        if (
           distance > wormhole_distance
        ):
            wormholes.append((edge,distance))
    return wormholes


if __name__ == "__main__":
    env = "Browntown"
    # Options for main map_type are
    # topological
    # pose
    G = main(env, map_type="topological")
#    pu.db
    d = get_dict(env)
    path = "../data/map/mapWorm20NewArch_" + env + "0.8.gpickle"
    G = nx.read_gpickle(path)
    print(len(list(G.nodes())))
    print(len(list(G.edges())))
    wormholes = find_wormholes(G, d, 0.80)
#    print(wormholes)
    sorted_by_second = sorted(wormholes, key=lambda tup: tup[1])
    print(sorted_by_second[-1])
#    neighbors = list(G.neighbors((1,60)))
#    print(neighbors)
#    slam_labels = torch.load("../data/results/slam/" + env + "/traj.pt")
#    d_slam = se3_to_habitat(slam_labels, d)
#    for n in neighbors:
#        node1 = (1,60)
#        node2 = n
#        position1 = d[node1[0]]["shortest_paths"][0][0][node1[1]]["position"]
#        position2 = d[node2[0]]["shortest_paths"][0][0][node2[1]]["position"]
#        distance = np.linalg.norm(np.array(position1) - np.array(position2))
#        print("****************")
#        print(node1)
#        print(node2)
#        print(distance)
#        print("****************")
#    for wormhole in wormholes:
#        print("****************")
#        print(wormhole[0][0])
#        print(list(G.neighbors(wormhole[0][0])))
#        print("****************")
