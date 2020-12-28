import numpy as np
from tqdm import tqdm
import habitat
import pudb
import networkx as nx
from make_map import get_dict
from habitat.datasets.utils import get_action_shortest_path


def create_sim(scene):
    cfg = habitat.get_config()
    cfg.defrost()
    cfg.SIMULATOR.SCENE = "../../data/scene_datasets/gibson/" + scene + ".glb"
    cfg.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR"]
    cfg.freeze()
    sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)
    return sim


# Returns the length of the shortest path between two edges
def actual_edge_len(edge, scene, d, sim):
    # Get starting position and rotation
    node1 = edge[0]
    pose1 = d[node1[0]]["shortest_paths"][0][node1[1]]
    position1 = pose1["position"]
    rotation1 = pose1["rotation"]
    # Get ending position
    node2 = edge[1]
    pose2 = d[node2[0]]["shortest_paths"][0][node2[1]]
    position2 = pose2["position"]
    rotation2 = pose2["rotation"]
    # Get actual distance from simulator, use same settings as the one used in data collection
    if position1 == position2:
        return 0
    shortest_path_success_distance = 0.2
    shortest_path_max_steps = 500
    shortest_path = get_action_shortest_path(
        sim,
        source_position=position1,
        source_rotation=rotation1,
        goal_position=position2,
        success_distance=shortest_path_success_distance,
        max_episode_steps=shortest_path_max_steps,
    )
    return len(shortest_path)


def main():
    test_envs = np.load("./model/test_env.npy")
    scene = test_envs[0]
    sim = create_sim(scene)
    d = get_dict(scene)
    G = nx.read_gpickle("../../data/map/map_Goodwine.gpickle")
    lengths = {}
    # For now only search over edges that go between different trajectories
    # The following will search over all edges
    # for edge in tqdm(list(G.edges)):
    graph_edges = list(G.edges)
    edges = np.array(list(G.edges))
    edges = edges.reshape(len(edges), 4)
    edges_idx = np.where(edges[:, 0] != edges[:, 2])[0]
    for idx in tqdm(edges_idx):
        edge = graph_edges[idx]
        length = actual_edge_len(edge, scene, d, sim)
        if length not in lengths:
            lengths[length] = 0
        lengths[length] += 1
    print(lengths)

    np.save("lengths2.npy", lengths)


if __name__ == "__main__":
    main()
