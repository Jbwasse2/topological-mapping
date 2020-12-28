import habitat
import numpy as np
import torch
from slam_agents import (
    ORBSLAM2Agent,
    get_config,
    cfg_baseline,
    make_good_config_for_orbslam2,
)
import argparse


def create_agent():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent-type",
        default="orbslam2-rgbd",
        choices=["blind", "orbslam2-rgbd", "orbslam2-rgb-monod"],
    )
    parser.add_argument("--task-config", type=str, default="tasks/pointnav_rgbd.yaml")
    args = parser.parse_args()

    config = habitat.get_config("./configs/tasks/pointnav_rgbd.yaml")
    agent_config = cfg_baseline()
    agent_config.defrost()
    config.defrost()
    config.ORBSLAM2 = agent_config.ORBSLAM2
    config.ORBSLAM2.SLAM_VOCAB_PATH = "./data/ORBvoc.txt"
    config.ORBSLAM2.SLAM_SETTINGS_PATH = "./data/mp3d3_small1k.yaml"
    config.SIMULATOR.SCENE = "../../data/scene_datasets/gibson/" + "Airport" + ".glb"
    make_good_config_for_orbslam2(config)

    if args.agent_type == "blind":
        agent = BlindAgent(config.ORBSLAM2)
    elif args.agent_type == "orbslam2-rgbd":
        agent = ORBSLAM2Agent(config.ORBSLAM2)
    elif args.agent_type == "orbslam2-rgb-monod":
        agent = ORBSLAM2MonodepthAgent(config.ORBSLAM2)
    else:
        raise ValueError(args.agent_type, "is unknown type of agent")
    return agent, config


def create_sim(scene, cfg):
    cfg.defrost()
    cfg.SIMULATOR.SCENE = "../../data/scene_datasets/gibson/" + scene + ".glb"
    cfg.freeze()
    sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)
    pu.db
    return sim


def main():
    agent, config = create_agent()
    print(config)
    pu.db
    sim = create_sim("Airport", config)
    #    sim.agents = [agent]
    habitat_observation = sim.step(2)
    k = habitat_observation.keys()
    pu.db
    agent.act(habitat_observation, random_prob=0.0)


if __name__ == "__main__":
    main()
