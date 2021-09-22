import math
import quaternion
import random
from pathlib import Path
import pandas as pd
import pickle
from glob import glob
import os
import time

import seaborn as sns
from matplotlib import rc
import matplotlib.patches as mpatches
import matplotlib
from scipy.spatial.transform import Rotation as R
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
from habitat.utils.geometry_utils import (
    angle_between_quaternions,
    quaternion_rotate_vector,
)
from habitat.tasks.utils import cartesian_to_polar
from make_map import (
    create_sim,
    get_dict,
    get_node_image,
    get_trajectory_env,
    get_true_label,
    se3_to_habitat,
    sparsify_trajectories,
)
from test_data import GibsonMapDataset
from worm_model.model import Siamese, SiameseDeepVO
from geoslam import get_slam_pose_labels

set_GPU = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = set_GPU
rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
# rc("text", usetex=True)
sns.set(style="darkgrid", font_scale=1.9)


def get_node_image_sequence(node, scene, max_lengths, transform=False, context=10):
    ret = []
    trnsform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ]
    )
    # Probably wanna cache this later...
    if node[0] not in max_lengths:
        max_length = len(
            glob(
                (
                    "../../data/datasets/pointnav/gibson/v4/train_large/images/"
                    + scene
                    + "/"
                    + "episodeRGB"
                    + str(node[0])
                    + "_*.jpg"
                )
            )
        )
        max_lengths[node[0]] = max_length
    else:
        max_length = max_lengths[node[0]]
    for i in range(node[1] - context + 1, node[1] + 1):
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
        image = cv2.resize(image, (128, 128)) / 255
        image = image - 0.5
        if transform:
            ret.append(trnsform(image))
        else:
            ret.append(image)
    return torch.stack(ret), max_lengths


def estimate_edge_len_wormhole(model, edges, sim, device, max_lengths, buffer_size):
    if edges == []:
        return [], [], max_lengths
    # Pre make the images or else for some reason pytorch allocates too much mem...
    images1 = torch.zeros(buffer_size, 10, 3, 128, 128)
    images2 = torch.zeros(buffer_size, 10, 3, 128, 128)
    for count, edge in enumerate(edges):
        scene = os.path.basename(sim.config.sim_cfg.scene.id).split(".")[0]
        images1[count], max_lengths = get_node_image_sequence(
            edge[0], scene, transform=True, max_lengths=max_lengths
        )
        images2[count], max_lengths = get_node_image_sequence(
            edge[1], scene, transform=True, max_lengths=max_lengths
        )
    images1 = (images1).float().to(device)
    images2 = (images2).float().to(device)
    images1 = images1[0: len(edges)]
    images2 = images2[0: len(edges)]
    if len(images1.shape) == 4:
        images1 = images1.unsqueeze(0)
        images2 = images2.unsqueeze(0)
    hidden = model.init_hidden(images1.shape[0], model.hidden_size, device)
    model.hidden = hidden
    pose, similarity = model(images1, images2)
    similarity = F.softmax(similarity)
    return (
        similarity[:, 1].detach().cpu().numpy(),
        pose.detach().cpu().numpy(),
        max_lengths,
    )


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
def find_close_node(
    model, edges, sim, device, similarity, map_type, max_lengths, buffer_size
):
    return_node = None
    if map_type == "topological":
        results, pose, max_lengths = estimate_edge_len_wormhole(
            model, edges, sim, device, max_lengths, buffer_size
        )
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
def find_close_nodes(
    model, edges, sim, device, similarity, map_type, max_lengths, buffer_size
):
    results, pose, max_lengths = estimate_edge_len_wormhole(
        model, edges, sim, device, max_lengths, buffer_size
    )
    return_edges = []
    for counter, result in enumerate(results):
        if result >= similarity:
            return_edges.append(counter)
    return return_edges, max_lengths


def add_trajs_to_graph_wormhole(
    G,
    trajectories,
    traj_ind,
    model,
    device,
    similarity,
    episodes,
    sim,
    d,
    buffer_size=200,
    map_type="topological",
    long_traj=True,
):
    assert long_traj == False or long_traj == True
    max_lengths = {}
    tail_node = None
    if map_type == "orbslamRGB" or map_type == "similarity_orbslamRGB":
        scene = os.path.basename(sim.config.sim_cfg.scene.id).split(".")[0]
        slam_pose_labels = get_slam_pose_labels(
            scene, len(trajectories), "mono", sim)
    elif map_type == "orbslamRGBD" or map_type == "similarity_orbslamRGBD":
        scene = os.path.basename(sim.config.sim_cfg.scene.id).split(".")[0]
        slam_pose_labels = get_slam_pose_labels(
            scene, len(trajectories), "orbslam2-rgbd", sim
        )
    print("PUT TQDM HERE")
    for trajectory_count, (trajectory_count_episode, trajectory) in enumerate(
        zip(episodes, trajectories)
    ):
        if not long_traj:
            tail_node = None
        print("PUT TQDM HERE")
        for frame_count, frame in enumerate(trajectory):
            # we want to keep track of what the actual frame is from in the trajectory
            true_label = get_true_label(
                trajectory_count, frame_count, traj_ind)
            current_node = (trajectory_count_episode, true_label)
            nodes = list(G.nodes)
            edges = []
            gt_offset = None
            if frame_count % 25 == 0:
                if (
                    map_type == "orbslamRGB"
                    or map_type == "orbslamRGBD"
                    or map_type == "similarity_orbslamRGBD"
                    or map_type == "similarity_orbslamRGB"
                ):
                    # https://github.com/Jbwasse2/topological-mapping/blob/f155aa133568306d7c72fc296810c6e4ac207506/src/slam/visualize_map.py
                    trajectory_point = slam_pose_labels[current_node]
                    if trajectory_point is None:
                        print("OH STAGAHSTS")
                        tail_node = None
                        current_node = None
                        continue
                    trajectory_point = np.array(
                        trajectory_point[1:]).reshape(3, 4)
                    position = trajectory_point[:, 3]
                    rot_se3 = trajectory_point[0:3, 0:3]
                    rot = R.from_matrix(rot_se3)
                    _, _, heading = rot.as_euler("xyz")
                    position = np.array(
                        [position[2], position[0], position[1]])
                    pose = np.append(position, heading).tolist()
                    current_node = (
                        current_node[0],
                        current_node[1],
                        None,
                        tuple(pose),
                    )
                    G.add_node(current_node)
                    if tail_node is not None:
                        G.add_edge(tail_node, current_node)

                elif (
                    tail_node != None
                    and map_type != "orbslamRGB"
                    and map_type != "orbslamRGBD"
                ):
                    if map_type != "similarity":
                        edges.append((tail_node, current_node))
                        (prob_similar, pose, max_lengths,) = estimate_edge_len_wormhole(
                            model,
                            edges,
                            sim,
                            device,
                            max_lengths,
                            buffer_size=len(edges),
                        )

                        #                        print("*****************")
                        #                        print(frame_count)
                        #                        print(prob_similar)
                        print(pose[0][0], pose[0][1])
                        #                        print("*****************")
                        pose = pose[0]
                        r = pose[0]
                        th = pose[1]
                        tail_node_global = tail_node[3]
                        theta_prev = tail_node_global[3]
                        # http://motion.cs.illinois.edu/RoboticSystems/Kinematics.html
                        global_pose = (
                            -r * math.cos(theta_prev + th) +
                                          tail_node_global[0],
                            r * math.sin(theta_prev + th) +
                                         tail_node_global[1],
                            0,
                            (th + tail_node_global[3]),
                        )
                        current_node = (
                            current_node[0],
                            current_node[1],
                            tuple(pose),
                            global_pose,
                        )
                        G.add_node(current_node)
                        G.add_edge(tail_node, current_node)
                    elif map_type == "similarity":
                        pose = (0, 0, 0, 0)
                        global_pose = (0, 0, 0, 0)
                        current_node = (
                            current_node[0],
                            current_node[1],
                            tuple(pose),
                            global_pose,
                        )
                        G.add_node(current_node)
                        G.add_edge(tail_node, current_node)

                else:
                    current_node = (
                        current_node[0],
                        current_node[1],
                        (0.0, 0.0, 0.0, 0.0),
                        (0.0, 0.0, 0.0, 0.0),
                    )
                    G.add_node(current_node)
                tail_node = current_node
    return G


# This function attempts to merge nodes together while putting them in the graph
# Overall I found this does not improve speed, but I am keeping this here for now...


def add_trajs_to_graph_wormhole_attempt2(
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
    pruned_nodes = {}
    for i in range(len(trajectories)):
        pruned_nodes[i] = 0
    for trajectory_count, (trajectory_count_episode, trajectory) in tqdm(
        enumerate(zip(episodes, trajectories))
    ):
        tail_node = None
        for frame_count, frame in tqdm(enumerate(trajectory)):
            # we want to keep track of what the actual frame is from in the trajectory
            true_label = get_true_label(
                trajectory_count, frame_count, traj_ind)
            current_node = (trajectory_count_episode, true_label)
            nodes = list(G.nodes)
            edges = []
            dont_add_flag = False
            for i in range(len(nodes)):
                node = nodes[i]
                if node != tail_node:
                    edge = (current_node, node)
                    edges.append(edge)
                # We use a list of edges to feed into the NN because batching is faster.
                # Check to see if there is already a similar enough edge in the graph
                if (
                    len(edges) >= buffer_size or i == len(nodes) - 1
                ):  # -2 because range is over n-1 and then one node should be tail node
                    results, max_lengths = find_close_node(
                        model,
                        edges,
                        sim,
                        device,
                        similarity,
                        map_type,
                        max_lengths,
                        buffer_size=buffer_size,
                    )
                    print(prob_similar)
                    if results == None:
                        pass
                    else:
                        current_node = edges[results][1]
                        pruned_nodes[trajectory_count] += 1
                        dont_add_flag = True
                        break
                    edges = []
            # If
            if tail_node is not None and not dont_add_flag:
                edges = []
                edges.append((tail_node, current_node))
                prob_similar, pose, max_lengths = estimate_edge_len_wormhole(
                    model, edges, sim, device, max_lengths, buffer_size=len(
                        edges)
                )
                pose = pose[0]
                tail_node_global = tail_node[3]
                global_pose = (
                    pose[0] + tail_node_global[0],
                    pose[1] + tail_node_global[1],
                    pose[2] + tail_node_global[2],
                    (pose[3] + tail_node_global[3] % np.pi),
                )
                current_node = (
                    current_node[0],
                    current_node[1],
                    tuple(pose),
                    global_pose,
                )
                G.add_node(current_node)
                G.add_edge(tail_node, current_node)
            elif tail_node is not None and dont_add_flag:
                G.add_edge(tail_node, current_node)
                dont_add_flag = False
            else:
                current_node = (
                    current_node[0],
                    current_node[1],
                    (0.0, 0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0, 0.0),
                )
                G.add_node(current_node)
            tail_node = current_node
        nx.write_gpickle(
            G,
            "./results/save_model/mapWorm20NewArch_"
            + str(trajectory_count)
            + ".gpickle",
        )
    print("Number of nodes pruned = ")
    print(pruned_nodes)
    return G


def create_topological_map_wormhole(
    trajectories,
    traj_ind,
    model,
    device,
    episodes,
    similarityNodes,
    similarityEdges,
    long_traj,
    sim=None,
    d=None,
    closeness=None,
    map_type="topological",
    scene=None,
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
        d=d,
        long_traj=long_traj,
    )

    #    print("Dont forget after loading in test G pickle")
    #    G = nx.read_gpickle("./test_G.gpickle")
    wormholes = find_wormholes(G, d, 5.0, visualize=True)
    G = connect_graph_trajectories_wormholes(
        G,
        model,
        device,
        similarityEdges,
        long_traj=long_traj,
        sim=sim,
        map_type=map_type,
        closeness=closeness,
    )

    return G


def connect_graph_trajectories_wormholes(
    G,
    model,
    device,
    similarity,
    long_traj,
    closeness,
    sim=None,
    buffer_size=200,
    map_type="topological",
):
    counter = 0
    nodes = list(G.nodes)
    max_lengths = {}
    distance_constraint_map_types = [
        "topological",
        "VO",
        "orbslamRGB",
        "orbslamRGBD",
        "similarity_orbslamRGB",
        "similarity_orbslamRGBD",
    ]
    visual_constraint_map_types = [
        "similarity",
        "topological",
        "similarity_orbslamRGB",
        "similarity_orbslamRGBD",
    ]
    if not long_traj:
        pu.db
    for node_i in tqdm(nodes):
        edges = []
        for i, node_j in enumerate(list(G.nodes)):
            # Only add nodes to edges if the predicted pose is close...
            # All of the map types we will be testing except similarity will be using pose
            if map_type in visual_constraint_map_types:
                if map_type in distance_constraint_map_types:
                    distance = np.linalg.norm(
                        np.array(node_i[3]) - np.array(node_j[3]))
                    if distance <= closeness and node_i != node_j:
                        edge = (node_i, node_j)
                        edges.append(edge)
                else:
                    edge = (node_i, node_j)
                    edges.append(edge)
                if (
                    len(edges) >= buffer_size or i == len(nodes) - 1
                ):  # -2 because range is over n-1 and then one node should be tail node
                    print(len(edges))
                    results, max_lengths = find_close_nodes(
                        model,
                        edges,
                        sim,
                        device,
                        similarity,
                        map_type=map_type,
                        max_lengths=max_lengths,
                        buffer_size=buffer_size,
                    )
                    for result in results:
                        edge = edges[result]
                        if edge[0] == edge[1]:
                            continue
                        G.add_edge(edge[0], edge[1])
                    edges = []
            elif map_type in distance_constraint_map_types:
                distance = np.linalg.norm(
                    np.array(node_i[3]) - np.array(node_j[3]))
                if distance <= closeness and node_i != node_j:
                    G.add_edge(node_i, node_j)
    return G


def main(
    env,
    closeness,
    similarityNodes=0.99,
    similarityEdges=0.80,
    map_type="topological",
    long_traj=True,
):
    seed = 0
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cuda:0")
    test_envs = np.load("./worm_model/test_env.npy")
    sim = create_sim(env)
    sparsifier_model = SiameseDeepVO().to(device)
    sparsifier_model.load_state_dict(
        torch.load("./worm_model/saved_model.pth"))
    sparsifier_model.eval()
    data = GibsonMapDataset(test_envs)
    # trajs is the trajectory of the 224x224 dataset, not sparsified
    trajs = get_trajectory_env(data, env, number_of_trajectories=2)
    # traj_new is sparsified trajectory with the traj_visual dataset
    # traj_ind says which indices where used
    traj_new, traj_ind = sparsify_trajectories(
        trajs,
        device,
        sim,
        sparsity=1,
    )
    #    print("DONT FORGET TO CLEAR TRAJ HERE")
    #    fp = open("traj_new.pkl", "rb")
    #    traj_new = pickle.load(fp)
    #    fp = open("traj_ind.pkl", "rb")
    #    traj_ind = pickle.load(fp)
    #    print("DONT FORGET TO TURN OFF TRAJ")
    episodes = list(range(len(traj_new)))
    G = create_topological_map_wormhole(
        traj_new,
        traj_ind,
        sparsifier_model,
        episodes=episodes,
        similarityNodes=similarityNodes,
        similarityEdges=similarityEdges,
        closeness=closeness,
        sim=sim,
        scene=env,
        device=device,
        map_type=map_type,
        long_traj=long_traj,
    )
    Path("../../data/map/" + str(map_type)).mkdir(parents=True, exist_ok=True)
    nx.write_gpickle(
        G,
        "../../data/map/"
        + str(map_type)
        + "/map50Worm"
        + str(len(traj_new))
        + "NewArchTest_"
        + str(env)
        + str(similarityEdges)
        + "_"
        + str(closeness)
        + ".gpickle",
    )
    return G


def visualize_map(G, d):
    def plotLine(p1, p2, color):
        x = (p1[0], p2[0])  # Extracting x's values of points
        y = (p1[1], p2[1])  # Extracting y's values of points
        plt.plot(x, y, "-o", color=color)  # Plotting points

    def coordinates2tuple(x, y):
        return (x, y)

    color_palette = sns.color_palette("colorblind")
    color1 = color_palette[3]
    color2 = color_palette[0]
    gt_offset = None
    gt_rot_offset = None
    for edge in list(G.edges()):
        node1, node2 = edge[0], edge[1]
        pu.db
        pose1 = d[node1[0]]["shortest_paths"][0][0][node1[1]]
        pose2 = d[node2[0]]["shortest_paths"][0][0][node2[1]]
        position1_gt = np.array(pose1["position"])
        position2_gt = np.array(pose2["position"])
        rot1_gt = quaternion.as_euler_angles(
            np.quaternion(*pose1["rotation"]))[1]
        rot2_gt = quaternion.as_euler_angles(
            np.quaternion(*pose2["rotation"]))[1]
        # Debugging...
        pos_goal = np.array(pose2["position"])
        pos_agent = np.array(pose1["position"])
        rot_agent = np.quaternion(*pose1["rotation"])
        direction_vector = pos_goal - pos_agent
        direction_vector_agent = quaternion_rotate_vector(
            rot_agent.inverse(), direction_vector
        )
        rho, phi = cartesian_to_polar(
            -direction_vector_agent[2], direction_vector_agent[0]
        )
        print(rho, phi)
        # End debug
        # Debug
        position1_pred = node1[3]
        position2_pred = node2[3]
        pu.db
        if gt_offset is None:
            gt_offset = position1_gt
            gt_rot_offset = -rot1_gt

        p1 = coordinates2tuple(
            (-position1_gt[2] + gt_offset[2]), position1_gt[0] - gt_offset[0]
        )
        p2 = coordinates2tuple(
            (-position2_gt[2] + gt_offset[2]), position2_gt[0] - gt_offset[0]
        )
        p1 = rotate_2d_pt(p1, gt_rot_offset)
        p2 = rotate_2d_pt(p2, gt_rot_offset)
        plotLine(p1, p2, color=color1)
        p1 = coordinates2tuple(position1_pred[0], position1_pred[1])
        p2 = coordinates2tuple(position2_pred[0], position2_pred[1])
        plotLine(p1, p2, color=color2)

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(18.5, 10.5)
    true_patch = mpatches.Patch(color=color1, label="Truth")
    est_patch = mpatches.Patch(color=color2, label="Prediction")
    plt.legend(handles=[true_patch, est_patch])
    plt.savefig("./results/map_visualize.pdf", bbox_inches="tight", dpi=300)


def rotate_2d_pt(pt, rot):
    x = pt[0] * np.cos(rot) - pt[1] * np.sin(rot)
    y = pt[0] * np.sin(rot) + pt[1] * np.cos(rot)
    return (x, y)


def rho_phi_to_2D_traj(rho, phi):
    poses = []
    for r, p in zip(rho, phi):
        prev_pose = poses[-1]
        theta_prev = poses[
        global_pose = (
            -r * math.cos(theta_prev + th) + tail_node_global[0],
            r * math.sin(theta_prev + th) + tail_node_global[1],
            0,
            (th + tail_node_global[3]),
        )
        current_node = (
            current_node[0],
            current_node[1],
            tuple(pose),
            global_pose,
        )


def find_wormholes(G, d, wormhole_distance=5.0, visualize=True):
    if visualize:
        visualize_map(G, d)
    wormholes = []
    distances = {
        "gt_x": [],
        "gt_y": [],
        "gt_z": [],
        "gt_r": [],
        "pred_x": [],
        "pred_y": [],
        "pred_z": [],
        "pred_r": [],
    }
    for edge in list(G.edges()):
        node1, node2 = edge[0], edge[1]
        pose1 = d[node1[0]]["shortest_paths"][0][0][node1[1]]
        pose2 = d[node2[0]]["shortest_paths"][0][0][node2[1]]
        position1 = pose1["position"]
        position2 = pose2["position"]
        rotDiff = angle_between_quaternions(
            np.quaternion(*pose1["rotation"]
                          ), np.quaternion(*pose2["rotation"])
        )
        distance = np.linalg.norm(np.array(position1) - np.array(position2))
        dist_vector_gt = tuple(np.array(position2) - np.array(position1))
        dist_vector_pred = node2[2]
        #        distances["gt_x"].append(dist_vector_gt[0])
        #        distances["gt_y"].append(dist_vector_gt[1])
        #        distances["gt_r"].append(rotDiff)
        #        distances["pred_x"].append(dist_vector_pred[0])
        #        distances["pred_y"].append(dist_vector_pred[1])
        #        distances["pred_r"].append(dist_vector_pred[3])
        if distance > wormhole_distance:
            wormholes.append((edge, distance))
    #    df = pd.DataFrame(distances)
    #    dfx = df[["gt_x", "pred_x"]]
    #    dfx.plot.bar()
    #    plt.savefig("./results/debugx.png")
    #    dfy = df[["gt_y", "pred_y"]]
    #    dfy.plot.bar()
    #    plt.savefig("./results/debugy.png")
    #    dfr = df[["gt_r", "pred_r"]]
    #    dfr.plot.bar()
    #    plt.savefig("./results/debugr.png")
    return wormholes


if __name__ == "__main__":
    start_time = time.time()
    env = "Browntown"
    map_type_test = "VO"
    long_traj = True
    possible_map_types = [
        "base",
        "topological",
        "VO",
        "similarity",
        "orbslamRGB",
        "orbslamRGBD",
        "similarity_orbslamRGB",
        "similarity_orbslamRGBD",
    ]
    assert map_type_test in possible_map_types
    if map_type_test in [
        "topological",
        "similarity_orbslamRGB",
        "similarity_orbslamRGBD",
    ]:
        test_similarityEdges = 0.90
        test_closeness = 1.25
    elif map_type_test in ["VO", "orbslamRGB", "orbslamRGBD"]:
        test_similarityEdges = None
        test_closeness = 1.0
    elif map_type_test in ["similarity"]:
        test_similarityEdges = 0.99
        test_closeness = None
    elif map_type_test in ["base"]:
        test_similarityEdges = None
        test_closeness = 0.0
    else:
        assert 1 == 0

    G = main(
        env,
        closeness=test_closeness,
        map_type=map_type_test,
        similarityEdges=test_similarityEdges,
        long_traj=long_traj,
    )
    d = get_dict(env)
    #    path = (
    #        "../data/map/" + map_type_test + "/mapWorm20NewArchDebug_" + env + "0.8.gpickle"
    #    )
    #    G = nx.read_gpickle(path)
    print(len(list(G.nodes())))
    print(len(list(G.edges())))
    #    if test_closeness == None or test_closeness == 0.0:
    #        test_closeness = 5.0
    #    wormholes = find_wormholes(G, d, 5.0, visualize=True)
    print(wormholes)
    print("Number of womrholes = " + str(len(wormholes)))
    print("Time to run = " + str(time.time() - start_time))
#    sorted_by_second = sorted(wormholes, key=lambda tup: tup[1])
#    print(sorted_by_second[-1])
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
