from data_getter import GibsonDataset
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import torch
import numpy as np
import pudb
from tqdm import tqdm
from torch.utils import data
from best_model.model import Siamese

sns.color_palette("flare", as_cmap=True)

dataset = GibsonDataset(
    "test",
    seed=0,
    samples=5000,
    max_distance=50,
    episodes=20,
    ignore_0=False,
    debug=False,
    give_distance=True,
)
test_dataloader = data.DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=10,
)
mse = nn.MSELoss()
device = torch.device("cuda:0")
test_envs = np.load("./best_model/test_env.npy")
sparsifier = Siamese().to(device)
sparsifier.load_state_dict(torch.load("./best_model/saved_model.pth"))
sparsifier.eval()
results = np.zeros((72, 2))
results_pose = {}
for i in range(72):
    results_pose[i] = []
for i, batch in enumerate(tqdm(test_dataloader)):
    (x, y, d, poseGT) = batch
    d[torch.where(d>=71)] = 70
    d[torch.where(d==-1)] = 71
    poseGT = poseGT.to(device)
    hidden = sparsifier.init_hidden(y.shape[0], sparsifier.hidden_size, device)
    sparsifier.hidden = hidden
    image1, image2 = x
    image1 = image1.to(device).float()
    image2 = image2.to(device).float()
    similarity, pose = sparsifier(image1, image2)
    #Handle distance estimation stuff
    for di, pGT, p in zip(d,poseGT, pose):
        results_pose[di.item()].append(torch.sqrt(mse(p, pGT)).detach().cpu().item())
    #Handle classifcaiton stuff
    result = torch.argmax(similarity, axis=1).detach().cpu()
    for di, r in zip(d, result):
        results[di][r] += 1
np.save("results.npy", results)
results = np.load("./results.npy")
np.save("resultsPose.npy", results_pose, allow_pickle=True)
results_pose = np.load("./resultsPose.npy", allow_pickle=True).item()
true = 0
total = 0
for i, row in enumerate(results):
    total += np.sum(row)
    if i >= 50:
        true += row[0]
    else:
        true += row[1]
print(true / total)
#Normalize by row
for counter, row in enumerate(results):
    s = np.sum(row)
    if s >= 0:
        results[counter] = row/s
ax = sns.heatmap(results, linewidths=0.1, annot=True)

fig = ax.get_figure()
fig.savefig("plotSim.png")
results = np.zeros((72,1))
plt.clf()
for row, key in results_pose.items():
    if len(key) > 0:
        results[row] = np.mean(key)
    else:
        results[row] = -1
plt.title("Average RMSE Pose error vs Distance")
plt.plot(results)

fig = ax.get_figure()
fig.savefig("plotPose.png")
