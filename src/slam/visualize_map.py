import torch
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
            #            if np.abs(position[1] - 0.07) <= 0.3:
            plt.plot(position[0], position[2], marker="o", color="r", ls="")
    plt.show()


def visualize_slam(slam_labels):
    device = torch.device("cuda:0")  # noqa: B008
    pu.db
    for counter, label in enumerate(slam_labels):
        se3 = homogenize_p(torch.from_numpy(label)[1:].view(3, 4).to(device)).view(
            1, 4, 4
        )
        position = np.array(se3[0][0:3, 3].tolist()).astype("float32")
        plt.plot(position[0], position[1], marker="o", color="r", ls="")
    plt.show()


if __name__ == "__main__":
    scene_name = "Bolton"
    ground_truth_d = get_dict(scene_name)
    data_dir = "../data/results/slam/" + scene_name + "/"
    slam_labels = torch.load(data_dir + "traj.pt")
    #    visualize_true(ground_truth_d)
    visualize_slam(slam_labels)
