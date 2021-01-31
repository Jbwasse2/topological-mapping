import numpy as np
import torch.nn.functional as F
import time
from tqdm import tqdm
import networkx as nx
import os
import matplotlib
import torch
import habitat
from make_map import (
    create_sim,
    get_trajectory_env,
    sparsify_trajectories,
    get_true_label,
    get_node_image,
    get_dict,
)
from test_data import GibsonMapDataset
from worm_model.model import Siamese
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
matplotlib.rcParams["font.family"] = "Helvetica"
font = {"weight": "bold"}


def estimate_edge_len_wormhole(model, edges, sim, device):
    # Get images for edges
    if edges == []:
        return []
    images1 = []
    images2 = []
    for edge in edges:
        scene = os.path.basename(sim.config.sim_cfg.scene.id).split(".")[0]
        image1 = get_node_image(edge[0], scene, transform=True)
        image2 = get_node_image(edge[1], scene, transform=True)
        images1.append(image1)
        images2.append(image2)
    images1 = torch.stack(images1).float().to(device)
    images2 = torch.stack(images2).float().to(device)
    results = F.softmax(model(images1, images2))
    return results[:, 1].detach().cpu().numpy()


# If a node in the graph is already similiar to the node being proposed, use the ndoe in graph isntead. This will just return 1 node that is close
def find_close_node(model, edges, sim, device, similarity):
    results = estimate_edge_len_wormhole(model, edges, sim, device)
    return_node = None
    for counter, result in enumerate(results):
        if result >= similarity:
            return_node = counter
            break
    return return_node


# This will return all nodes that are close
def find_close_nodes(model, edges, sim, device, similarity):
    results = estimate_edge_len_wormhole(model, edges, sim, device)
    return_edges = []
    for counter, result in enumerate(results):
        if result >= similarity:
            return_edges.append(counter)
    return return_edges


def add_trajs_to_graph_wormhole(
    G, trajectories, traj_ind, model, device, similarity, episodes, sim, buffer_size=400
):
    for trajectory_count, (trajectory_count_episode, trajectory) in tqdm(
        enumerate(zip(episodes, trajectories))
    ):
        tail_node = None
        for frame_count, frame in enumerate(trajectory):
            # we want to keep track of what the actual frame is from in the trajectory
            true_label = get_true_label(trajectory_count, frame_count, traj_ind)
            current_node = (trajectory_count_episode, true_label)
            nodes = list(G.nodes)
            edges = []
            for i in range(len(nodes)):
                # Skip previous tail node from being too close to current node, we already added this.
                node = nodes[i]
                edge = (current_node, node)
                edges.append(edge)
                if node == tail_node:
                    continue
                if (
                    len(edges) == buffer_size or i == len(nodes) - 2
                ):  # -2 because range is over n-1 and then one node should be tail node
                    results = find_close_node(model, edges, sim, device, similarity)
                    if results == None:
                        pass
                    else:
                        current_node = edges[results][1]
                        break
                    edges = []
            # Check to see if there is already a similar enough edge in the graph
            G.add_node(current_node)
            if tail_node is not None:
                G.add_edge(tail_node, current_node)
            else:
                pass
            tail_node = current_node
    return G


def create_topological_map_wormhole(
    trajectories,
    traj_ind,
    model,
    device,
    episodes,
    similarity,
    sim=None,
    d=None,
    scene=None,
):
    # Put all trajectories into graph
    G = nx.DiGraph()
    G = add_trajs_to_graph_wormhole(
        G,
        trajectories,
        traj_ind,
        model,
        device,
        similarity,
        episodes,
        sim=sim,
    )
    G = connect_graph_trajectories_wormholes(
        G,
        model,
        device,
        similarity,
        sim=sim,
    )

    return G


def connect_graph_trajectories_wormholes(
    G, model, device, similarity, sim=None, buffer_size=400
):
    counter = 0
    nodes = list(G.nodes)
    for node_i in tqdm(nodes):
        edges = []
        for i, node_j in enumerate(list(G.nodes)):
            edge = (node_i, node_j)
            edges.append(edge)
            if (
                len(edges) == buffer_size or i == len(nodes) - 1
            ):  # -2 because range is over n-1 and then one node should be tail node
                results = find_close_nodes(model, edges, sim, device, similarity)
                for result in results:
                    edge = edges[result]
                    if edge[0] == edge[1]:
                        continue
                    G.add_edge(edge[0], edge[1])
                edges = []
    return G


def main(env):
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
        sparsity=15,
    )
    episodes = list(range(len(traj_new)))
    G = create_topological_map_wormhole(
        traj_new,
        traj_ind,
        sparsifier_model,
        episodes=episodes,
        similarity=0.99,
        sim=sim,
        scene=env,
        device=device,
    )
    nx.write_gpickle(G, "../../data/map/mapWorm_" + str(env) + ".gpickle")
    return G


def find_wormholes(G, d, wormhole_distance=1.5):
    wormholes = []
    for edge in list(G.edges()):
        node1, node2 = edge[0], edge[1]
        position1 = d[node1[0]]["shortest_paths"][0][0][node1[0]]["position"]
        position2 = d[node2[0]]["shortest_paths"][0][0][node2[0]]["position"]
        if (
            np.linalg.norm(np.array(position1) - np.array(position2))
            > wormhole_distance
        ):
            wormholes.append(edge)
    return wormholes


if __name__ == "__main__":
    env = "Harkeyville"
    G = main(env)
    d = get_dict(env)
    print(len(list(G.nodes())))
    print(len(list(G.edges())))
    print(len(find_wormholes(G, d)))
