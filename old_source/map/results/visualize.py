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
        with open(path, "r+", encoding="utf-8") as f:
            info = f.readlines()
    except Exception as e:
        print("Failed to open " + path + " - " + str(e))
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
for f in glob.glob("./wormholes/*"):
    # Only check data files
    if os.path.splitext(f)[-1] != "":
        continue
    wormhole, edge, node = get_data(f)
    if wormhole == None:
        print("Failed ", f)
        continue
    names.append(os.path.basename(f))
    wormholes.append(wormhole)
    edges.append(edge)
    nodes.append(node)
wormholes_clean = []
edges_clean = []
nodes_clean = []
names_clean = []
for f in glob.glob("./wormholesClean/*"):
    # Only check data files
    if os.path.splitext(f)[-1] != "":
        continue
    wormhole, edge, node = get_data(f)
    if wormhole == None:
        print("Failed ", f)
        continue
    names_clean.append(os.path.basename(f))
    wormholes_clean.append(wormhole)
    edges_clean.append(edge)
    nodes_clean.append(node)
pu.db
df = pd.DataFrame({"edges": edges, "wormholes": wormholes}, index=names)
df_clean = pd.DataFrame(
    {"edges_clean": edges_clean, "wormholes_clean": wormholes_clean}, index=names_clean
)
df_2 = df.join(df_clean, how="outer")
df_2 = df_2[["edges", "wormholes", "edges_clean", "wormholes_clean"]]
ax = df_2.plot.bar(rot=0, figsize=(32, 8.5))
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
plt.title("Number of Edges and Wormholes in Topological Map vs Cleaned Topological Map")
plt.savefig("./edge_worm_vs_env.jpg", bbox_inches="tight")
plt.clf()
ax = df_2.mean().plot.bar(rot=0, figsize=(10, 10))
plt.title(
    "Average Number of Edges and Wormholes Across Topological Map and Cleaned Topological Map"
)
for p in ax.patches:
    ax.annotate(str(p.get_height())[0:5], (p.get_x() * 1.005, p.get_height() * 1.005))
plt.savefig("./average_worm_edge.jpg", bbox_inches="tight")
