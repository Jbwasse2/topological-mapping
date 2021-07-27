import gzip
import os
import random
import time
from copy import deepcopy
from typing import ClassVar, Dict, List

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pudb
import torch
import torchvision.transforms as transforms
from quaternion import as_euler_angles, quaternion
from tqdm import tqdm

import habitat
from data.results.sparsifier.best_model.model import Siamese
from habitat import Config, logger
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import (
    agent_state_target2ref,
    angle_between_quaternions,
    quaternion_rotate_vector,
)
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.config.default import get_config
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
from habitat_baselines.utils.common import poll_checkpoint_folder
from habitat_baselines.utils.env_utils import construct_envs

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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


def get_checkpoint(config):
    current_ckpt = None
    prev_ckpt_ind = -1
    while current_ckpt is None:
        current_ckpt = poll_checkpoint_folder(config.EVAL_CKPT_PATH_DIR, prev_ckpt_ind)
        time.sleep(0.1)  # sleep for 2 secs before polling again
    return current_ckpt


def get_ddppo_model(config, device):
    checkpoint = get_checkpoint(config)
    ckpt_dict = torch.load(checkpoint)
    trainer = PPOTrainer(config)
    config.defrost()
    config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
    config.freeze()
    ppo_cfg = config.RL.PPO
    trainer.envs = construct_envs(config, get_env_class(config.ENV_NAME))
    trainer.device = device
    trainer._setup_actor_critic_agent(ppo_cfg)
    trainer.agent.load_state_dict(ckpt_dict["state_dict"])
    actor_critic = trainer.agent.actor_critic
    actor_critic.eval()
    test_recurrent_hidden_states = torch.zeros(
        actor_critic.net.num_recurrent_layers,
        config.NUM_PROCESSES,
        ppo_cfg.hidden_size,
        device=device,
    )
    return actor_critic, test_recurrent_hidden_states


def create_sim(scene):
    cfg = habitat.get_config("../../configs/tasks/pointnav_gibson.yaml")
    cfg.defrost()
    cfg.SIMULATOR.SCENE = "../../data/scene_datasets/gibson/" + scene + ".glb"
    cfg.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
    cfg.freeze()
    sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)
    return sim


def example_forward(model, hidden_state, scene, device):
    ob = scene.reset()
    prev_action = torch.zeros(1, 1).to(device)
    not_done_masks = torch.zeros(1, 1).to(device)
    not_done_masks += 1
    ob["pointgoal_with_gps_compass"] = torch.rand(1, 2).to(device)
    ob["depth"] = torch.from_numpy(ob["depth"]).unsqueeze(0).to(device)
    model.act(ob, hidden_state, prev_action, not_done_masks, deterministic=False)


# Takes a node such as (0,5) and reutrns its heading/position in the collected trajectory
def get_node_pose(node, d):
    try:
        pose = d[node[0]][node[1]]
        position = pose["position"]
        rotation = pose["rotation"]
    except KeyError:
        try:
            pose = d[node[0]]["shortest_paths"][0][0][node[1]]
        except Exception as e:
            pu.db
            print("FUCK")
        position = pose["position"]
        rotation = pose["rotation"]
    return position, rotation


def try_to_reach(
    G,
    start_node,
    end_node,
    d,
    ddppo_model,
    localization_model,
    hidden_state,
    sim,
    device,
    visualize=True,
):
    ob = sim.reset()
    # Perform high level planning over graph
    if visualize:
        video_name = "local.mkv"
        video = cv2.VideoWriter(video_name, 0, 3, (256 * 2, 256 * 1))
    else:
        video = None
    try:
        path = nx.dijkstra_path(G, start_node, end_node)
    except nx.exception.NetworkXNoPath as e:
        return 3
    if len(path) <= 3:
        return 4
    print("NEW PATH")
    current_node = path[0]
    local_goal = path[1]
    # Move robot to starting position/heading
    agent_state = sim.agents[0].get_state()
    ground_truth_d = get_dict("Browntown")
    pos, rot = get_node_pose(current_node, ground_truth_d)
    agent_state.position = pos
    agent_state.rotation = rot
    sim.agents[0].set_state(agent_state)
    # Start experiments!
    for current_node, local_goal in zip(path, path[1:]):
        success = try_to_reach_local(
            current_node,
            local_goal,
            d,
            ddppo_model,
            localization_model,
            hidden_state,
            sim,
            device,
            video,
        )
        if success != 1:
            if visualize:
                cv2.destroyAllWindows()
                video.release()
            return 1
    if visualize:
        cv2.destroyAllWindows()
        video.release()
    # Check to see if agent made it
    agent_pos = sim.agents[0].get_state().position
    ground_truth_d = get_dict("Browntown")
    try:
        (episode, frame, local_pose, global_pose) = end_node
    except Exception as e:
        print(e)
        pu.db
    goal_pos = ground_truth_d[episode]["shortest_paths"][0][0][frame]["position"]
    distance = np.linalg.norm(agent_pos - goal_pos)
    if distance >= 0.2:
        print(distance)
        return 2
    return 0


def get_node_depth(node, scene_name):
    image_location = (
        "../../data/datasets/pointnav/gibson/v4/train_large/images/"
        + scene_name
        + "/"
        + "episodeDepth"
        + str(node[0])
        + "_"
        + str(node[1]).zfill(5)
        + ".jpg"
    )
    return plt.imread(image_location)


def get_node_image(node, scene_name):
    image_location = (
        "../../data/datasets/pointnav/gibson/v4/train_large/images/"
        + scene_name
        + "/"
        + "episodeRGB"
        + str(node[0])
        + "_"
        + str(node[1]).zfill(5)
        + ".jpg"
    )
    return plt.imread(image_location)


# Returns 1 on success, and 0 or -1 on failure
def try_to_reach_local(
    start_node,
    local_goal_node,
    d,
    ddppo_model,
    localization_model,
    hidden_state,
    sim,
    device,
    video,
):
    MAX_NUMBER_OF_STEPS = 200
    prev_action = torch.zeros(1, 1).to(device)
    not_done_masks = torch.zeros(1, 1).to(device)
    not_done_masks += 1
    ob = sim.get_observations_at(sim.get_agent_state())
    actions = []
    # Double check this is right RGB
    # goal_image is for video/visualization
    # goal_image_model is for torch model for predicting distance/heading
    scene_name = os.path.splitext(os.path.basename(sim.config.sim_cfg.scene.id))[0]
    if video is not None:
        scene_name = os.path.splitext(os.path.basename(sim.config.sim_cfg.scene.id))[0]

        #        start_image = cv2.resize(get_node_image(start_node, scene_name), (256, 256))
        goal_image = cv2.resize(get_node_image(local_goal_node, scene_name), (256, 256))

    for i in range(MAX_NUMBER_OF_STEPS):
        displacement = torch.from_numpy(
            get_displacement_local_goal(sim, local_goal_node, d)
        ).type(torch.float32)

        if video is not None:
            image = np.hstack([ob["rgb"], goal_image])
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.putText(
                image,
                text=str(displacement).replace("tensor(", "").replace(")", ""),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                org=(10, 10),
                fontScale=0.5,
                color=(255, 255, 255),
                thickness=2,
            )
            video.write(image)
        ob["pointgoal_with_gps_compass"] = displacement.unsqueeze(0).to(device)
        ob["depth"] = torch.from_numpy(ob["depth"]).unsqueeze(0).to(device)
        with torch.no_grad():
            _, action, _, hidden_state = ddppo_model.act(
                ob, hidden_state, prev_action, not_done_masks, deterministic=False
            )
        actions.append(action[0].item())
        prev_action = action
        if action[0].item() == 0:  # This is stop action
            print("STOP ACTION")
            return 1
        if displacement[0] < 0.2:
            print("STOP GT")
            return 1
        ob = sim.step(action[0].item())
    return 0


def get_displacement_local_goal(sim, local_goal, d):
    # See https://github.com/facebookresearch/habitat-lab/blob/b7a93bc493f7fb89e5bf30b40be204ff7b5570d7/habitat/tasks/nav/nav.py
    # for more information
    pos_goal, rot_goal = get_node_pose(local_goal, d)
    # Quaternion is returned as list, need to change datatype
    if isinstance(rot_goal, type(np.array)):
        rot_goal = quaternion(*rot_goal)
    pos_agent = sim.get_agent_state().position
    rot_agent = sim.get_agent_state().rotation
    direction_vector = pos_goal - pos_agent
    direction_vector_agent = quaternion_rotate_vector(
        rot_agent.inverse(), direction_vector
    )
    rho, phi = cartesian_to_polar(-direction_vector_agent[2], direction_vector_agent[0])

    # Should be same as agent_world_angle
    return np.array([rho, -phi])


def visualize_observation(observation, start, goal):
    f, ax = plt.subplots(1, 3)
    ax[0].imshow(observation["rgb"])
    ax[0].title.set_text("Current Observation")
    ax[1].imshow(start)
    ax[1].title.set_text("Local Start Image")
    ax[2].imshow(goal)
    ax[2].title.set_text("Local Goal Image")
    plt.show()


def run_experiment(
    G, d, ddppo_model, localization_model, hidden_state, scene, device, experiments=100
):
    # Choose 2 random nodes in graph
    # 0 means success
    # 1 means failed at runtime
    # 2 means the system thought it finished the path, but was actually far away
    # 3 means topological map failed to find a path
    # 4 shouldn't be returned as an experiment result, its just used to say that the path in the map is trivially easy (few nodes to traverse between).
    return_codes = [0 for i in range(4)]
    for _ in tqdm(range(experiments)):
        results = None
        while results == None or results == 4:
            node1, node2 = get_two_nodes(G)
            results = try_to_reach(
                G,
                node1,
                node2,
                d,
                ddppo_model,
                localization_model,
                deepcopy(hidden_state),
                scene,
                device,
            )
        return_codes[results] += 1

    return return_codes


# Currently just return 2 random nodes, in the future may do something smarter.
def get_two_nodes(G):
    return random.sample(list(G.nodes()), 2)


def get_localization_model(device):
    model = Siamese().to(device)
    model.load_state_dict(
        torch.load("./data/results/sparsifier/best_model/saved_model.pth")
    )
    model.eval()
    return model


def main():
    env = "Browntown"
    G = nx.read_gpickle("./data/map/mapWorm20NewArch_" + env + "0.8.gpickle")
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    config = get_config("configs/baselines/ddppo_pointnav.yaml", [])
    device = (
        torch.device("cuda", config.TORCH_GPU_ID)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    localization_model = get_localization_model(device)
    scene = create_sim(env)
    ddppo_model, hidden_state = get_ddppo_model(config, device)
    # example_forward(model, hidden_state, scene, device)
    # d = np.load("../data/map/d_slam.npy", allow_pickle=True).item()
    d = get_dict(env)
    results = run_experiment(
        G, d, ddppo_model, localization_model, hidden_state, scene, device
    )
    print(results)


if __name__ == "__main__":
    main()
