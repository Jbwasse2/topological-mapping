import matplotlib.pyplot as plt
import rclpy
import pudb
import argparse
import numpy as np
import seaborn as sns
from matplotlib import rc
from top_map.topological_map import (
    TopologicalMap,
)
import networkx as nx

# Plot nodes/edges


# https://stackoverflow.com/questions/53967392/creating-a-graph-with-images-as-nodes
# Will need to play around with piesize and arrowsize to make this work...
def graph_visualize(top_map):
    piesize = 0.11  # this is the image size
    p2 = piesize / 2.0
    G = top_map.map
    images = top_map.meng
    fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot(111)
    ax.set_aspect("equal")
    pos = nx.circular_layout(G)
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=100)

    trans = ax.transData.transform
    trans2 = fig.transFigure.inverted().transform
    for n in G:
        xx, yy = trans(pos[n])  # figure coordinates
        xa, ya = trans2((xx, yy))  # axes coordinates
        a = plt.axes([xa - p2, ya - p2, piesize, piesize])
        a.set_aspect("equal")
        image = images[n]
        a.imshow(image)
        a.axis("off")
    plt.show()


rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
rc("text", usetex=True)
sns.set(style="darkgrid", font_scale=1.9)

parser = argparse.ArgumentParser("topological_map_visualization")
parser.add_argument("pkl", help="topological map pkl location", type=str)
args = parser.parse_args()
rclpy.init()
topMap = TopologicalMap(wait_for_service=False)
topMap.load(args.pkl)
graph_visualize(topMap)
rclpy.shutdown()