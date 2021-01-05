import gzip
import os
import random
import time
from copy import deepcopy
from typing import ClassVar, Dict, List

import habitat
import networkx as nx
import numpy as np
import pudb
import torch
from habitat import Config, logger
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.config.default import get_config
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
from habitat_baselines.utils.common import poll_checkpoint_folder
from habitat_baselines.utils.env_utils import construct_envs


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
    pu.db
    prev_action = torch.zeros(1, 1).to(device)
    not_done_masks = torch.zeros(1, 1).to(device)
    ob["pointgoal_with_gps_compass"] = torch.rand(1, 2).to(device)
    ob["depth"] = torch.from_numpy(ob["depth"]).unsqueeze(0).to(device)
    model.act(ob, hidden_state, prev_action, not_done_masks, deterministic=False)


def try_to_reach(G, start_node, end_node, model, hidden_state, scene, device):
    # Perform high level planning over graph
    try:
        path = nx.dijkstra_path(G, start_node, end_node)
    except nx.exception.NetworkXNoPath as e:
        return 2
    pu.db
    # Move robot to starting position/heading
    current_node = path[0]
    local_goal = path[1]
    for i in range(len(path) - 1):
        success = try_to_reach_local(
            current_node, local_goal, model, hidden_state, scene, device
        )
        if not success:
            return 1
        else:
            current_node = local_goal
            local_goal = path[i + 2]
    return 0


def try_to_reach_local(start_node, local_goal_node, model, hidden_state, scene, device):
    MAX_NUMBER_OF_STEPS = 50
    prev_action = torch.zeros(1, 1).to(device)
    not_done_masks = torch.zeros(1, 1).to(device)
    ob = scene.reset()
    for i in range(MAX_NUMBER_OF_STEPS):
        pass


def visualize_observation(observation):
    pass


def run_experiment(G, model, hidden_state, scene, device, experiments=100):
    # Choose 2 random nodes in graph
    return_codes = [0 for i in range(3)]
    for _ in range(experiments):
        node1, node2 = get_two_nodes(G)
        # Hard code for now...
        node1 = (13, 10)
        node2 = (31, 60)
        results = try_to_reach(
            G, node1, node2, model, deepcopy(hidden_state), scene, device
        )
        return_codes[results] += 1
    return return_codes


# Currently just return 2 random nodes, in the future may do something smarter.
def get_two_nodes(G):
    return random.sample(list(G.nodes()), 2)


def main():
    random.seed(0)
    config = get_config("configs/baselines/ddppo_pointnav.yaml", [])
    device = (
        torch.device("cuda", config.TORCH_GPU_ID)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    scene = create_sim("Goodwine")
    #  model, hidden_state = get_ddppo_model(config, device)
    #  example_forward(model, hidden_state, scene, device)
    G = nx.read_gpickle("./data/map/map_Goodwine.gpickle")
    d = get_dict("Goodwine")
    #    traj_ind = np.load("./traj_ind.npy", allow_pickle=True)
    #    traj_ind_eval = np.load("./traj_ind_eval.npy", allow_pickle=True)
    #    traj_new = np.load("./traj_new.npy", allow_pickle=True)
    #    traj_new_eval = np.load("./traj_new_eval.npy", allow_pickle=True)
    #    eval_trajs = np.load("./eval_trajs.npy", allow_pickle=True)
    run_experiment(G, None, None, scene, device)
    pu.db


if __name__ == "__main__":
    main()
