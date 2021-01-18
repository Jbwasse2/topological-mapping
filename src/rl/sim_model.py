import gzip
import os
import random
import time
from copy import deepcopy
from typing import ClassVar, Dict, List

import cv2
import habitat
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pudb
import torch
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
from quaternion import as_euler_angles, quaternion
from tqdm import tqdm


def get_dict(fname):
    f = gzip.open(
        "../../data/datasets/pointnav/gibson/v3/train_large/content/"
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
    pose = d[node[0]]["shortest_paths"][0][0][node[1]]
    position = pose["position"]
    rotation = pose["rotation"]
    return position, rotation


def try_to_reach(
    G, start_node, end_node, d, model, hidden_state, sim, device, visualize=True
):
    ob = sim.reset()
    # Perform high level planning over graph
    if visualize:
        video_name = "local.mkv"
        video = cv2.VideoWriter(video_name, 0, 3, (256, 256 * 3))
    else:
        video = None
    try:
        path = nx.dijkstra_path(G, start_node, end_node)
    except nx.exception.NetworkXNoPath as e:
        return 2
    print("NEW PATH")
    current_node = path[0]
    local_goal = path[1]
    # Move robot to starting position/heading
    agent_state = sim.agents[0].get_state()
    pos, rot = get_node_pose(current_node, d)
    agent_state.position = pos
    agent_state.rotation = rot
    sim.agents[0].set_state(agent_state)
    # Start experiments!
    for current_node, local_goal in zip(path, path[1:]):
        success = try_to_reach_local(
            current_node, local_goal, d, model, hidden_state, sim, device, video
        )
        if success != 1:
            if visualize:
                cv2.destroyAllWindows()
                video.release()
            return 1
    if visualize:
        cv2.destroyAllWindows()
        video.release()
    return 0


def get_node_image(node, scene_name):
    image_location = (
        "../../data/datasets/pointnav/gibson/v3/train_large/images/"
        + scene_name
        + "/"
        + "episode"
        + str(node[0])
        + "_"
        + str(node[1]).zfill(5)
        + ".jpg"
    )
    return plt.imread(image_location)


# Returns 1 on success, and 0 or -1 on failure
def try_to_reach_local(
    start_node, local_goal_node, d, model, hidden_state, sim, device, video
):
    MAX_NUMBER_OF_STEPS = 200
    prev_action = torch.zeros(1, 1).to(device)
    not_done_masks = torch.zeros(1, 1).to(device)
    not_done_masks += 1
    ob = sim.get_observations_at(sim.get_agent_state())
    actions = []
    if video is not None:
        scene_name = os.path.splitext(os.path.basename(sim.config.sim_cfg.scene.id))[0]

        start_image = cv2.resize(get_node_image(start_node, scene_name), (256, 256))
        goal_image = cv2.resize(get_node_image(local_goal_node, scene_name), (256, 256))
        image = np.vstack([ob["rgb"], start_image, goal_image])
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        video.write(image)

    for i in range(MAX_NUMBER_OF_STEPS):
        displacement = torch.from_numpy(
            get_displacement_local_goal(sim, local_goal_node, d)
        ).type(torch.float32)
        ob["pointgoal_with_gps_compass"] = displacement.unsqueeze(0).to(device)
        ob["depth"] = torch.from_numpy(ob["depth"]).unsqueeze(0).to(device)
        with torch.no_grad():
            _, action, _, hidden_state = model.act(
                ob, hidden_state, prev_action, not_done_masks, deterministic=False
            )
        actions.append(action[0].item())
        prev_action = action
        if action[0].item() == 0 or displacement[0] < 0.2:  # This is stop action
            print("STOP")
            return 1
        ##        elif action[0].item() == 0 and displacement[0] >= 0.2:
        ##            pu.db
        ##            print("STOP")
        ##            return -1
        if video is not None:
            image = np.vstack([ob["rgb"], start_image, goal_image])
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.putText(
                image,
                text=str(displacement).replace("tensor(", "").replace(")", ""),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                org=(10, 260),
                fontScale=0.5,
                color=(255, 255, 255),
                thickness=2,
            )
            video.write(image)
        ob = sim.step(action[0].item())
    return 0


def angle_between_vectors(v1, v2):
    return np.arccos((v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def get_displacement_local_goal(sim, local_goal, d):
    # See https://github.com/facebookresearch/habitat-lab/blob/b7a93bc493f7fb89e5bf30b40be204ff7b5570d7/habitat/tasks/nav/nav.py
    # for more information
    pos_goal, rot_goal = get_node_pose(local_goal, d)
    # Quaternion is returned as list, need to change datatype
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


def run_experiment(G, d, model, hidden_state, scene, device, experiments=100):
    # Choose 2 random nodes in graph
    # 0 means success
    # 1 means failed at runtime
    # 2 means no path found
    return_codes = [0 for i in range(3)]
    for _ in tqdm(range(experiments)):
        node1, node2 = get_two_nodes(G)
        node1, node2 = (3, 28), (0, 48)
        results = try_to_reach(
            G, node1, node2, d, model, deepcopy(hidden_state), scene, device
        )
        if results == 1:
            print(node1, node2)
        return_codes[results] += 1
        break
    return return_codes


# Currently just return 2 random nodes, in the future may do something smarter.
def get_two_nodes(G):
    return random.sample(list(G.nodes()), 2)


def main():
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    config = get_config("configs/baselines/ddppo_pointnav.yaml", [])
    device = (
        torch.device("cuda", config.TORCH_GPU_ID)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    scene = create_sim("Goodwine")
    model, hidden_state = get_ddppo_model(config, device)
    # example_forward(model, hidden_state, scene, device)
    G = nx.read_gpickle("./data/map/map_Goodwine0.05.gpickle")

    d = get_dict("Goodwine")
    #    traj_ind = np.load("./traj_ind.npy", allow_pickle=True)
    #    traj_ind_eval = np.load("./traj_ind_eval.npy", allow_pickle=True)
    #    traj_new = np.load("./traj_new.npy", allow_pickle=True)
    #    traj_new_eval = np.load("./traj_new_eval.npy", allow_pickle=True)
    #    eval_trajs = np.load("./eval_trajs.npy", allow_pickle=True)
    results = run_experiment(G, d, model, hidden_state, scene, device)
    print(results)


if __name__ == "__main__":
    main()
