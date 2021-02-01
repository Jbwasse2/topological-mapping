import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import pudb
import pandas as pd

sns.set(style="darkgrid", font_scale=1.3)
matplotlib.rcParams["font.family"] = "Helvetica"
font = {"weight": "bold"}

matplotlib.rc("font", **font)

# There exists a race condition with the simulator and printing the results to the screen, so getting the data might be tricky...
def get_data(path):
    try:
        with open(path, "r") as f:
            info = f.readlines()
    except Exception as e:
        pu.db
    try:
        # Case where results prints first and then simulator prints stuff to screen
        wormholes = int(info[-9])
        edges = int(info[-10])
        nodes = int(info[-11])
    except Exception as e:
        try:
            # Case where simulator prints to screen first then results get printed
            wormholes = int(info[-1])
            edges = int(info[-2])
            nodes = int(info[-3])
        except Exception as e:
            return None, None, None
    return wormholes, edges, nodes


wormholes = []
edges = []
nodes = []
names = []
for f in glob.glob("./*"):
    # Only check data files
    if os.path.splitext(f)[-1] != "":
        continue
    wormhole, edge, node = get_data(f)
    if wormhole == None:
        print("Failed ", f)
        continue
    names.append(f[2:])
    wormholes.append(wormhole)
    edges.append(edge)
    nodes.append(node)
df = pd.DataFrame({"edges": edges, "wormholes": wormholes}, index=names)
ax = df.plot.bar(rot=0, figsize=(17, 8.5))
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
plt.savefig("./edge_worm_vs_env.jpg", bbox_inches="tight")
plt.clf()
ax = df.mean().plot.bar(rot=0, figsize=(10, 10))
plt.savefig("./average_worm_edge.jpg", bbox_inches="tight")
