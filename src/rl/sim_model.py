import pudb
import habitat
from habitat_baselines.utils.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class
import os
import time
from typing import ClassVar, Dict, List

import torch

from habitat import Config, logger
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import poll_checkpoint_folder
from habitat_baselines.config.default import get_config
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer


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


def main():
    config = get_config("configs/baselines/ddppo_pointnav.yaml", [])
    device = (
        torch.device("cuda", config.TORCH_GPU_ID)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    scene = create_sim("Airport")
    model, hidden_state = get_ddppo_model(config, device)
    example_forward(model, hidden_state, scene, device)


if __name__ == "__main__":
    main()
