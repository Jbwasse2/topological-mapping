import argparse
import glob
import gzip
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pudb
import scipy.misc
from PIL import Image
from tqdm import tqdm

import habitat

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def get_dict(fname):
    f = gzip.open(
        "./data/datasets/pointnav/gibson/v2/train_large/content/" + fname + ".json.gz"
    )
    content = f.read()
    content = content.decode()
    content = content.replace("null", "None")
    content = eval(content)
    return content["episodes"]


def create_sim(scene):
    cfg = habitat.get_config()
    cfg.defrost()
    cfg.SIMULATOR.SCENE = "./data/scene_datasets/gibson/" + scene + ".glb"
    cfg.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
    cfg.SIMULATOR.RGB_SENSOR.WIDTH = 256
    cfg.SIMULATOR.RGB_SENSOR.HEIGHT = 256
    cfg.SIMULATOR.DEPTH_SENSOR.WIDTH = 256
    cfg.SIMULATOR.DEPTH_SENSOR.HEIGHT = 256
    cfg.freeze()
    sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)
    return sim


def generate_image_dataset(scene):
    sim = create_sim(scene)
    image_path = "./data/datasets/pointnav/gibson/v2/train_large/images/" + scene + "/"
    Path(image_path).mkdir(parents=True, exist_ok=True)
    d = get_dict(scene)
    # for collection in range(len(d)):
    for collection in tqdm(range(20)):
        episdoe_dict = d[collection]
        counter = 0
        for pose in episdoe_dict["shortest_paths"][0]:
            position = pose["position"]
            rotation = pose["rotation"]
            results = sim.get_observations_at(position, rotation)
            image = results["rgb"]
            matplotlib.image.imsave(
                image_path
                + "episodeRGB"
                + str(collection)
                + "_"
                + str(counter).zfill(5)
                + ".jpg",
                image,
            )
            image_depth = results["depth"]
            np.save(
                image_path
                + "episodeDepth"
                + str(collection)
                + "_"
                + str(counter).zfill(5)
                + ".npy",
                image_depth,
            )
            counter += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env_name", help="No", required=True)
    args = parser.parse_args()
    generate_image_dataset(args.env_name)
