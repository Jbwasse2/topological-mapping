import gzip
from torchvision import transforms as transforms
import itertools
import math
import os
import pathlib
import random
import shutil

import cv2
import habitat
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pudb
import quaternion
import seaborn as sns
import torch
import torch.nn.functional as F
from habitat.datasets.utils import get_action_shortest_path
from habitat.utils.geometry_utils import (
    angle_between_quaternions,
    quaternion_from_coeff,
    quaternion_to_list,
)
from habitat_baselines.slambased.reprojection import homogenize_p
from tqdm import tqdm
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import angle_between_quaternions

# from model.model import Siamese
from test_data import GibsonMapDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
matplotlib.rcParams["font.family"] = "Helvetica"
font = {"weight": "bold"}

matplotlib.rc("font", **font)


def create_sim(scene):
    cfg = create_cfg(scene)
    sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)
    return sim


def create_cfg(scene):
    cfg = habitat.get_config("../../configs/tasks/pointnav_gibson.yaml")
    cfg.defrost()
    cfg.SIMULATOR.SCENE = "../../data/scene_datasets/gibson/" + scene + ".glb"
    cfg.DATASET.SCENES_DIR = "../../data/scene_datasets/"
    cfg.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR"]
    cfg.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    cfg.freeze()
    return cfg


# Returns the length of the shortest path between two edges
def actual_edge_len(edge, d, sim):
    # Get starting position and rotation
    node1 = edge[0]
    pose1 = d[node1[0]]["shortest_paths"][0][0][node1[1]]
    position1 = pose1["position"]
    rotation1 = pose1["rotation"]
    # Get ending position
    node2 = edge[1]
    pose2 = d[node2[0]]["shortest_paths"][0][0][node2[1]]
    position2 = pose2["position"]
    rotation2 = pose2["rotation"]
    # Get actual distance from simulator, use same settings as the one used in data collection
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


# Full calculations and sim of gt length is expensive, only care about close points, so estimate of point may be sufficient
def estimate_edge_len(edge, d, sim):
    # Get starting position and rotation
    POSITION_GRANULARITY = 0.25  # meters
    ANGLE_GRANULARITY = math.radians(10)  # 10 degrees in radians
    node1 = edge[0]
    pose1 = d[node1[0]]["shortest_paths"][0][0][node1[1]]
    position1 = pose1["position"]
    rotation1 = quaternion.quaternion(*pose1["rotation"])
    # Get ending position
    node2 = edge[1]
    pose2 = d[node2[0]]["shortest_paths"][0][0][node2[1]]
    position2 = pose2["position"]
    rotation2 = quaternion.quaternion(*pose2["rotation"])
    # Get actual distance from simulator, use same settings as the one used in data collection
    angle = angle_between_quaternions(rotation1, rotation2)
    distance = np.linalg.norm(np.array(position1) - np.array(position2))

    return int(distance / POSITION_GRANULARITY) + int(angle / ANGLE_GRANULARITY)


# Oautput of ORB SLAM is SE3 matrix
# Habitat expects quaternion and position
# rotation is quaternion and position is list
def se3_to_habitat(data, d):
    counter = 0
    d_ret = {}
    for traj_count, traj in enumerate(d):
        d_ret[traj_count] = {}
        for frame_count, frame in enumerate(traj["shortest_paths"][0][0]):
            if data[counter] is None:
                d_ret[traj_count][frame_count] = None
                counter += 1
                continue
            foo = torch.from_numpy(data[counter])[1:].view(3, 4)
            se3 = homogenize_p(foo).view(4, 4)
            rotation_matrix = se3[0:3, 0:3]
            rotation_quaternion = quaternion.from_rotation_matrix(rotation_matrix)
            position = np.array(se3[0:3, 3].tolist()).astype("float32")
            # Something is wrong here i think, it should be len(3) but it may be len 4
            counter += 1
            local_pose = {"position": position, "rotation": rotation_quaternion}
            d_ret[traj_count][frame_count] = local_pose
    return d_ret


def estimate_edge_len_SLAM(edge, d, sim):
    POSITION_GRANULARITY = 0.25 / 5  # meters
    ANGLE_GRANULARITY = math.radians(10 / 5)

    node1 = edge[0]
    pose1 = d[node1[0]][node1[1]]
    position1 = pose1["position"]
    rotation1 = pose1["rotation"]
    # Get ending position
    node2 = edge[1]
    pose2 = d[node2[0]][node2[1]]
    position2 = pose2["position"]
    rotation2 = pose2["rotation"]
    angle = angle_between_quaternions(rotation1, rotation2)
    distance = np.linalg.norm(np.array(position1) - np.array(position2))

    return int(distance / POSITION_GRANULARITY) + int(angle / ANGLE_GRANULARITY)


def get_dict(fname):
    f = gzip.open(
        "../../data/datasets/pointnav/gibson/v4/train_large/content/"
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
def sparsify_trajectories(trajs, device, sim, sparsity):
    traj_ind = []
    counter = 0
    for traj in tqdm(trajs):
        counter += 1
        image_indices = []
        for i in range(len(traj)):
            if i % sparsity == 0:
                image_indices.append(i)
        traj_ind.append(image_indices)
    ret = []
    for traj, indices in tqdm(zip(trajs, traj_ind)):
        ret.append([traj[i] for i in indices])
    return ret, traj_ind


# Gets all of the trajectories in the environment
def get_trajectory_env(data, env, number_of_trajectories=None):
    data.set_env(env)
    if number_of_trajectories == None:
        number_of_trajectories = data.number_of_trajectories
    counter = 0
    trajs = []
    for i in tqdm(range(number_of_trajectories)):
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
def get_node_image(node, scene, transform=False):
    image_location = (
        "../../data/datasets/pointnav/gibson/v4/train_large/images/"
        + scene
        + "/"
        + "episodeRGB"
        + str(node[0])
        + "_"
        + str(node[1]).zfill(5)
        + ".jpg"
    )
    trnsform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = plt.imread(image_location)
    image = cv2.resize(image, (224, 224)) / 255
    if transform:
        return trnsform(image)
    else:
        return image


def get_true_label(trajectory_count, frame_count, traj_ind):
    try:
        return traj_ind[trajectory_count][frame_count]
    except Exception as e:
        return None


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
            if true_label == None:
                tail_node = None
                current_node = None
                continue
            current_node = (trajectory_count_episode, true_label)
            for node in list(G.nodes):
                # Skip previous tail node from being too close to current node, otherwise you may just be sparsifying your graph extra.
                # if node == tail_node:
                #    continue
                edge = (current_node, node)
                results = estimate_edge_len_SLAM(edge, d, sim)
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
    counter = 0
    for node_i in tqdm(list(G.nodes)):
        for node_j in list(G.nodes):
            if node_i[0] == node_j[0]:
                continue
            edge = (node_i, node_j)
            node1 = edge[0]
            pose1 = d[node1[0]][node1[1]]
            position1 = pose1["position"]
            # Get ending position
            node2 = edge[1]
            pose2 = d[node2[0]][node2[1]]
            position2 = pose2["position"]
            results = estimate_edge_len_SLAM(edge, d, sim)
            # Make sure local goal position is in front of local start position
            rho, phi = get_displacement_label(node1, node2, d)
            if results <= similarity and np.abs(phi) < 1.57:  # approx 90 degrees
                # Don't add edge if nodes are on seperate floors
                if abs(position1[1] - position2[1]) > 0.1:
                    continue
                G.add_edge(node_i, node_j)
    return G


def get_node_pose(node, d):
    pose = d[node[0]][node[1]]
    position = pose["position"]
    rotation = pose["rotation"]
    return position, rotation


def get_displacement_label(local_start, local_goal, d):
    # See https://github.com/facebookresearch/habitat-lab/blob/b7a93bc493f7fb89e5bf30b40be204ff7b5570d7/habitat/tasks/nav/nav.py
    # for more information
    pos_start, rot_start = get_node_pose(local_start, d)
    pos_goal, rot_goal = get_node_pose(local_goal, d)
    # Quaternion is returned as list, need to change datatype
    direction_vector = pos_goal - pos_start
    direction_vector_start = quaternion_rotate_vector(
        rot_start.inverse(), direction_vector
    )
    rho, phi = cartesian_to_polar(-direction_vector_start[2], direction_vector_start[0])

    # Should be same as agent_world_angle
    return np.array([rho, -phi])


def remove_bad_SLAM_labels(traj_ind, d_slam):
    for episode in range(len(d_slam)):
        for traj in range(len(d_slam[episode])):
            if d_slam[episode][traj] == None:
                if traj in traj_ind[episode]:
                    traj_ind[episode].remove(traj)

    return traj_ind


def d_to_new_format(d):
    ret = {}
    for traj_count, traj in enumerate(d):
        ret[traj_count] = {}
        for frame_count, frame in enumerate(traj["shortest_paths"][0][0]):
            position = np.array(frame["position"])
            rotation_quaternion = quaternion.quaternion(*frame["rotation"])
            local_pose = {"position": position, "rotation": rotation_quaternion}
            ret[traj_count][frame_count] = local_pose
    return ret


def create_topological_map(
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
    # Get labels from SLAM
    env = os.path.basename(sim.config.sim_cfg.scene.id).split(".")[0]
    slam_labels = np.array(torch.load("../data/results/slam/" + env + "/traj.pt"))
    total_trajs = 0
    # Make sure number of labels == number of trajs
    for i in range(len(d)):
        total_trajs += len(d[i]["shortest_paths"][0][0])
    assert slam_labels.shape[0] == total_trajs
    d_slam = se3_to_habitat(slam_labels, d)
    d = d_to_new_format(d)
    np.save("../data/map/d_slam.npy", d_slam)
    # Remove any nodes that have a None label from SLAM
    traj_ind = remove_bad_SLAM_labels(traj_ind, d_slam)
    G = add_trajs_to_graph(
        G,
        trajectories,
        traj_ind,
        model,
        device,
        similarity,
        episodes,
        sim=sim,
        d=d_slam,
    )
    G = connect_graph_trajectories(
        G,
        trajectories,
        traj_ind,
        model,
        device,
        similarity,
        episodes,
        sim=sim,
        d=d_slam,
    )
    return G


def build_graph(hold_out_percent, env):
    VISUALIZE = False
    CREATE_TRAJECTORIES = True
    device = torch.device("cuda:0")
    #    model = Siamese().to(device)
    #    model.load_state_dict(torch.load("./model/saved_model.pth"))
    #    model.eval()
    test_envs = np.load("./model/test_env.npy")
    d = get_dict(env)
    sim = create_sim(env)

    if CREATE_TRAJECTORIES == True:
        data = GibsonMapDataset(test_envs)
        # trajs is the trajectory of the 224x224 dataset, not sparsified
        trajs = get_trajectory_env(data, env)
        # traj_new is sparsified trajectory with the traj_visual dataset
        # traj_ind says which indices where used
        traj_new, traj_ind = sparsify_trajectories(
            trajs,
            device,
            sim,
            sparsity=1,
        )
        np.save("traj_new.npy", traj_new)
        np.save("traj_ind.npy", traj_ind)
        np.save("trajs.npy", trajs)
    else:
        traj_new = np.load("traj_new.npy", allow_pickle=True)
        traj_ind = np.load("traj_ind.npy", allow_pickle=True)
    if VISUALIZE:
        visualize_traj_ind(traj_ind)
    # some trajectories will be used to build the map, some will be used to evaluate the failure of imagegoal (Nevermind, I am not actually doing this), set hold_out to 0.0%.
    traj_held_out = int(len(traj_new) * hold_out_percent)
    eval_trajs = random.sample(list(range(len(traj_ind))), k=traj_held_out)
    map_trajs = list(set(range(len(traj_ind))) - set(eval_trajs))
    pu.db

    traj_new_map = [traj_new[i] for i in map_trajs]
    traj_ind_map = [traj_ind[i] for i in map_trajs]
    traj_new_eval = [traj_new[i] for i in eval_trajs]
    traj_ind_eval = [traj_ind[i] for i in eval_trajs]

    G = create_topological_map(
        traj_new_map,
        traj_ind_map,
        model,
        device,
        episodes=map_trajs,
        similarity=1,
        sim=sim,
        d=d,
        scene=env,
    )

    nx.write_gpickle(
        G, "../../data/map/map_" + str(env) + str(hold_out_percent) + ".gpickle"
    )
    return G, traj_new_eval, traj_ind_eval


def find_wormholes(G, world, similarity):
    # Want to go over trajectories, get their image and find the reachability
    traj_new = np.load("traj_new.npy", allow_pickle=True)
    traj_ind_global = np.load("traj_ind.npy", allow_pickle=True)
    trajectories = np.load("traj_new_eval.npy", allow_pickle=True)[0:2]
    traj_ind = np.load("traj_ind_eval.npy", allow_pickle=True)[0:2]
    device = torch.device("cuda:0")
    #    model = Siamese().to(device)
    #    model.load_state_dict(torch.load("./model/saved_model.pth"))
    #    model.eval()
    test_envs = np.load("./model/test_env.npy")
    d = get_dict(world)
    sim = create_sim(world)
    img_dir = "../../data/results/map/edges/"
    shutil.rmtree(img_dir)
    pathlib.Path(img_dir).mkdir(parents=True, exist_ok=False)
    counter = 0
    episodes = np.load("eval_trajs.npy", allow_pickle=True)
    for trajectory_count, (trajectory_count_episode, trajectory) in tqdm(
        enumerate(zip(episodes, trajectories))
    ):
        for frame_count, frame in enumerate(trajectory):
            print(trajectory_count)
            image1 = frame.unsqueeze(0).float().to(device)
            # we want to keep track of what the actual frame is from in the trajectory
            true_label = get_true_label(trajectory_count, frame_count, traj_ind)
            current_node = (trajectory_count_episode, true_label)
            for node in list(G.nodes):
                node_traj_ind = traj_ind_global[node[0]]
                node_ind = node_traj_ind.index(node[1])
                # Get original node image
                node_image = traj_new[node[0]][node_ind]
                image2 = node_image.unsqueeze(0).float().to(device)
                results = np.argmax(model(image1, image2).cpu().detach().numpy())
                if results in range(5, 5 + similarity):
                    print("Accepted")
                    edge = (current_node, node)
                    true_length = actual_edge_len(edge, d, sim)
                    visualize_edge(
                        current_node,
                        node,
                        world,
                        img_dir
                        + "edge_"
                        + str(counter).zfill(4)
                        + "_pred_"
                        + str(results)
                        + "_true_"
                        + str(true_length)
                        + ".png",
                    )
                    counter += 1
                else:
                    print("Passed")


def visualize_graph(env_name, G):
    cfg = create_cfg(env_name)
    env = habitat.Env(config=cfg)


if __name__ == "__main__":
    import random

    random.seed(0)
    env = "Bolton"
    G, traj_new_eval, traj_ind_eval = build_graph(0.00, env)
# G = nx.read_gpickle("../../data/map/map_Goodwine.gpickle")
# test_envs = np.load("./model/test_env.npy")
# ENV = test_envs[0]
# visualize_graph(ENV, G)
# wormholes = find_wormholes(G, ENV)
