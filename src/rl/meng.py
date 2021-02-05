import glob
import os
import random

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

import habitat
from habitat_baselines.config.default import get_config
from rmp_nav.simulation import agent_factory, sim_renderer
from sim_model import get_dict, get_node_image, get_node_pose
from topological_nav.reachability import model_factory
from topological_nav.reachability.planning import NavGraph, NavGraphSPTM


def get_node_image_goal(node, scene_name, neighbors=5):
    image_location_pre =  "../../data/datasets/pointnav/gibson/v4/train_large/images/" + scene_name + "/" + "episodeRGB" + str(node[0]) + "_"
    images_in_traj =glob.glob(image_location_pre + "*")
    number_of_images_in_traj = len(images_in_traj)
    assert node[1] < number_of_images_in_traj
    image_indices_to_return = []
    for i in range(node[1] - neighbors, node[1] + neighbors+ 1):
        if i < 0:
            image_indices_to_return.append(0)
        elif i >= number_of_images_in_traj:
            image_indices_to_return.append(number_of_images_in_traj-1)
        else:
            image_indices_to_return.append(i)
    images_ret = []
    for i in image_indices_to_return:
        image_location = (
            image_location_pre
            + str(i).zfill(5)
            + ".jpg"
        )
        image = plt.imread(image_location)
        images_ret.append(image)
    return np.array(images_ret)

def create_sim(scene):
    cfg = habitat.get_config("../../configs/tasks/pointnav_gibson.yaml")
    cfg.defrost()
    cfg.SIMULATOR.SCENE = "../../data/scene_datasets/gibson/" + scene + ".glb"
    cfg.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
    cfg.freeze()
    sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)
    return sim

class GetDistance:
    def __init__(self):
        self.model = self.get_model()

    def get_wp(self, model, ob, goal):
        agent = agent_factory.agents_dict[model["agent"]]()
        pu.db
        follower = model["follower"]
        return (
            follower.motion_policy.predict_waypoint(ob, goal),
       #     follower.sparsifier.predict_reachability(ob, goal),
        )

    def get_model(self):
        model = model_factory.get("model_12env_v2_future_pair_proximity_z0228")(
            device="cpu"
        )
        return model
    # Cv2 gives images in BGR, and from 0-255
    # We want RGB and from 0-1
    # Can also get list/ np array of images, this should be handled
    def cv2_to_model_im(self, im):
        im = np.asarray(im)
        assert len(im.shape) == 3 or len(im.shape) == 4
        if len(im.shape) == 3:
            im = np.swapaxes(im, 0, 2)
            im = np.swapaxes(im, 1, 2)
            im = np.asarray(im)
            im = (im / 255).astype("float32")
        else:
            im = np.swapaxes(im, 1, 3)
            im = np.swapaxes(im, 2, 3)
            im = np.asarray(im)
            im = (im / 255).astype("float32")
        return im
    def get_distances_test(self):
        ob = np.random.rand(3, 64,64)
        goal = np.random.rand(11, 3, 64,64)
        wp = self.get_wp(self.model, ob, goal)
        return wp

def try_to_reach(
    G,
    start_node,
    end_node,
    d,
    localization_model,
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
    if len(path) <= 10:
        return 4
    print("NEW PATH")
    current_node = path[0]
    local_goal = path[1]
    # Move robot to starting position/heading
    agent_state = sim.agents[0].get_state()
    ground_truth_d = get_dict("Bolton")
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
            localization_model,
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
    ground_truth_d = get_dict("Bolton")
    (episode, frame) = end_node
    goal_pos = ground_truth_d[episode]["shortest_paths"][0][0][frame]["position"]
    distance = np.linalg.norm(agent_pos - goal_pos)
    if distance >= 0.2:
        return 2
    return 0


def try_to_reach_local(
    start_node,
    local_goal_node,
    d,
    localization_model,
    sim,
    device,
    video,
):
    MAX_NUMBER_OF_STEPS = 100
    prev_action = torch.zeros(1, 1).to(device)
    not_done_masks = torch.zeros(1, 1).to(device)
    not_done_masks += 1
    ob = sim.get_observations_at(sim.get_agent_state())
    actions = []
    # Double check this is right RGB
    # goal_image is for video/visualization
    # goal_image_model is for torch model for predicting distance/heading
    scene_name = os.path.splitext(os.path.basename(sim.config.sim_cfg.scene.id))[0]
    goal_image_model = (
        cv2.resize(get_node_image(local_goal_node, scene_name), (224, 224)) / 255
    )
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    goal_image_model = (
        transform(goal_image_model).to(device).unsqueeze(0).type(torch.float32)
    )
    if video is not None:
        scene_name = os.path.splitext(os.path.basename(sim.config.sim_cfg.scene.id))[0]

        #        start_image = cv2.resize(get_node_image(start_node, scene_name), (256, 256))
        goal_image = cv2.resize(get_node_image(local_goal_node, scene_name), (256, 256))

    for i in range(MAX_NUMBER_OF_STEPS):
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
        ob["depth"] = torch.from_numpy(ob["depth"]).unsqueeze(0).to(device)
        action = None
        actions.append(action[0].item())
        prev_action = action
        if displacement[0] < 0.1:  # This is stop action
            print("STOP")
            return 1
        ob = sim.step(action[0].item())
    return 0

def run_experiment(
    G, d, localization_model, scene, device, experiments=100
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
                localization_model,
                scene,
                device,
            )
        return_codes[results] += 1
        if results == 2:
            pu.db
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
            break
    return return_codes


# Currently just return 2 random nodes, in the future may do something smarter.
def get_two_nodes(G):
    return random.sample(list(G.nodes()), 2)


def main():
    seed = 4
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    getDistance = GetDistance()
    config = get_config("configs/baselines/ddppo_pointnav.yaml", [])
    device = (
        torch.device("cuda", config.TORCH_GPU_ID)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    scene = create_sim("Bolton")
    
    # example_forward(model, hidden_state, scene, device)
    G = nx.read_gpickle("./data/map/map_Bolton0.0.gpickle")
    d = np.load("../data/map/d_slam.npy", allow_pickle=True).item()
    results = run_experiment(
        G, d, getDistance, scene, device
    )
    print(results)


if __name__ == "__main__":
    main()
    dist = GetDistance()
    print(dist.get_distances_test())
    #x = -1, y= -0
