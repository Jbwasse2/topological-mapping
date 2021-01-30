import torch
import quaternion
import networkx as nx
import numpy as np
import pudb
import matplotlib.pyplot as plt
from main import get_dict
from tqdm import tqdm
from habitat_baselines.slambased.reprojection import homogenize_p


def visualize_true(d):
    NUMBER_OF_TRAJECTORIES_COLLECTED = 20
    for i in tqdm(range(NUMBER_OF_TRAJECTORIES_COLLECTED)):
        for j in range(len(d[i]["shortest_paths"][0][0])):
            position = d[i]["shortest_paths"][0][0][j]["position"]
            plt.plot(position[0], position[2], marker="o", color="r", ls="")
    plt.show()


def visualize_slam(slam_labels):
    device = torch.device("cuda:0")  # noqa: B008
    for counter, label in tqdm(enumerate(slam_labels)):
        try:
            se3 = homogenize_p(torch.from_numpy(label)[1:].view(3, 4).to(device)).view(
                1, 4, 4
            )
            position = np.array(se3[0][0:3, 3].tolist()).astype("float32")
            plt.plot(position[2], position[0], marker="o", color="b", ls="")
        except Exception as e:
            pass
    plt.show()


def project_slam_on_map(slam_labels, obstacles):
    pass


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


def project_map_on_traj(slam_labels, G, ground_truth_d):
    d_slam = se3_to_habitat(slam_labels, ground_truth_d)
    device = torch.device("cuda:0")  # noqa: B008
    pu.db
    for counter, label in tqdm(enumerate(slam_labels)):
        # It is OK to skip nodes here if the actual SLAM pose is None
        # This visualizes ALL of the labels, so this makes sense
        try:
            se3 = homogenize_p(torch.from_numpy(label)[1:].view(3, 4).to(device)).view(
                1, 4, 4
            )
            position = np.array(se3[0][0:3, 3].tolist()).astype("float32")
            plt.plot(position[2], position[0], marker="o", color="b", ls="")
        except Exception as e:
            pass
    # It is NOT OK to skip nodes here if the SLAM pose is NONE here because
    # that should have been filtered out while building the map!
    for node in list(G.nodes()):
        position = d_slam[node[0]][node[1]]["position"]
        plt.plot(position[2], position[0], marker="o", color="r", ls="")

    plt.show()


if __name__ == "__main__":
    scene_name = "Bolton"
    ground_truth_d = get_dict(scene_name)
    data_dir = "../data/results/slam/" + scene_name + "/"
    slam_labels = torch.load(data_dir + "traj.pt")
    obstacles = torch.load(data_dir + "obstacles.pt")
    G = nx.read_gpickle("../data/map/map_Bolton0.0.gpickle")
    project_map_on_traj(slam_labels, G, ground_truth_d)

    # visualize_true(ground_truth_d)
    # visualize_slam(slam_labels)
