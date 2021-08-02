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
#from best_model.model import Siamese
from translation.model import Siamese
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sns.color_palette("flare", as_cmap=True)

dataset = GibsonDataset(
    "test",
    seed=0,
    samples=2000,
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
    num_workers=0,
)
mse = nn.MSELoss()
device = torch.device("cuda:0")
test_envs = np.load("./translation/test_env.npy")
sparsifier = Siamese().to(device)
sparsifier.load_state_dict(torch.load("./translation/saved_model.pth"))
sparsifier.eval()
results = np.zeros((72, 2))
results_pose_T = {}
results_pose_R = {}
for i in range(72):
    results_pose_T[i] = []
    results_pose_R[i] = []
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
    pose, similarity = sparsifier(image1, image2)
    T = torch.norm(poseGT[:,0:3], dim=1)
    #Handle distance estimation stuff
    for di, pGT, p in zip(d,T, pose):
        results_pose_T[di.item()].append(torch.sqrt(mse(p, pGT)).detach().cpu().item())
#        results_pose_R[di.item()].append(torch.sqrt(mse(p[3], pGT[3])).detach().cpu().item())
    #Handle classifcaiton stuff
    #result = torch.argmax(similarity, axis=1).detach().cpu()
#    for di, r in zip(d, result):
#        results[di][r] += 1
np.save("results.npy", results)
#np.save("resultsPoseR.npy", results_pose_R, allow_pickle=True)
np.save("resultsPoseT.npy", results_pose_T, allow_pickle=True)
results = np.load("./results.npy")
results_pose_T = np.load("./resultsPoseT.npy", allow_pickle=True).item()
#results_pose_R = np.load("./resultsPoseR.npy", allow_pickle=True).item()
ys = []
#true = 0
#total = 0
#for i, row in enumerate(results):
#    total += np.sum(row)
#    if i >= 50:
#        true += row[0]
#    else:
#        true += row[1]
#print(true / total)
##Normalize by row
#for counter, row in enumerate(results):
#    s = np.sum(row)
#    if s >= 0:
#        results[counter] = row/s
#ax = sns.heatmap(results, linewidths=0.1)
#
#fig = ax.get_figure()
#fig.savefig("plotSim.png")
results_T = np.zeros((72,1))
results_R = np.zeros((72,1))
plt.clf()
for row, key in results_pose_T.items():
    if len(key) > 0:
        results_T[row] = np.mean(key)
    else:
        results_T[row] = -1
#for row, key in results_pose_R.items():
#    if len(key) > 0:
#        results_R[row] = np.mean(key)
#    else:
#        results_R[row] = -1
plt.title("Average RMSE Pose error vs Distance")
plt.plot(results_T, label="Translation")
#plt.plot(results_R, label="Rotation")
plt.legend()

#fig = ax.get_figure()
plt.savefig("plotPose.png")
