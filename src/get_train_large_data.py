import glob
import gzip
import os
from pathlib import Path

import pudb

import habitat


def get_dict(fname):
    f = gzip.open("../data/datasets/pointnav/gibson/v2/train_large/content/" + fname)
    content = f.read()
    content = content.decode()
    content = content.replace("null", "None")
    content = eval(content)
    return content["episodes"]


# Scene is .glb in ./data/scene_datasets/gibson/Dansville.glb
def create_sim(scene):
    scene = "../data/scene_datasets/gibson/" + scene + ".glb"
    cfg = habitat.get_config()
    cfg.defrost()
    cfg.SIMULATOR.SCENE = scene
    cfg.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR"]
    cfg.freeze()

    sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)
    return sim


def generate_image_dataset():
    for path in Path("../data/datasets/pointnav/gibson/v2/train_large/content/").rglob(
        "*.gz"
    ):
        name = os.path.splitext(path.name.split(".")[0])[0]
        image_path = (
            "../data/datasets/pointnav/gibson/v2/train_large/images/" + name + "/"
        )
        Path(image_path).mkdir(parents=True, exist_ok=True)
        d = get_dict(path.name)
        sim = create_sim(name)
        print(dir(sim))
        print(type(sim))
        for episode in range(len(d)):
            episdoe_dict = d[episode]
            for pose in episdoe_dict["shortest_paths"][0]:
                position = pose["position"]
                foo = sim.get_observations_at(position)
                rotation = pose["rotation"]
                break
            break


if __name__ == "__main__":
    generate_image_dataset()
