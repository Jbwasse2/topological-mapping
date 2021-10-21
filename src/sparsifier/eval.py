from data_getter import GibsonDataset
import matplotlib
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import torch
import numpy as np
import pudb
from tqdm import tqdm
from torch.utils import data
from best_model.model import Siamese, SiameseDeepVO
import os
from matplotlib import rc

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
rc("text", usetex=True)
sns.set(style="darkgrid", font_scale=1.9)


# Make plots for loss
lR = np.load("./best_model/lossesR.npy")
lS = np.load("./best_model/lossesS.npy")
lT = np.load("./best_model/lossesT.npy")
lRv = np.load("./best_model/lossesR_v.npy")
lSv = np.load("./best_model/lossesS_v.npy")
lTv = np.load("./best_model/lossesT_v.npy")
l = np.load("./best_model/losses.npy")
lv = np.load("./best_model/losses_v.npy")
acc = np.load("./best_model/accuracy.npy")
accv = np.load("./best_model/accuracy_v.npy")
plt.plot(lR, label="RotationT")
plt.plot(lRv, label="RotationV")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.legend()
plt.title("Rotation Prediction Loss")
plt.savefig("./results/Rot_loss.png", bbox_inches="tight")
plt.clf()
plt.plot(lS, label="SimilarityT")
plt.plot(lSv, label="SimilarityV")
plt.xlabel("Epochs")
plt.ylabel("Cross Entropy Loss")
plt.legend()
plt.title("Similarity Prediction Loss")
plt.savefig("./results/Sim_loss.png", bbox_inches="tight")
plt.clf()
plt.plot(lT, label="TranslationT")
plt.plot(lTv, label="TranslationV")
plt.ylabel("MSE Loss")
plt.xlabel("Epochs")
plt.legend()
plt.title("Translation Prediction Loss")
plt.savefig("./results/Trans_loss.png", bbox_inches="tight")
plt.clf()
plt.plot(l, label="LossT")
plt.plot(lv, label="LossV")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.legend()
plt.title("Total Prediction Loss")
plt.savefig("./results/Total_loss.png", bbox_inches="tight")
plt.clf()
plt.plot(acc, label="AccT")
plt.plot(accv, label="AccV")
plt.xlabel("Epochs")
plt.ylabel("Classification Accuracy")
plt.legend()
plt.title("Similarity Accuracy")
plt.savefig("./results/Acc.png", bbox_inches="tight")
plt.clf()
assert 1 == 0


sns.color_palette("flare", as_cmap=True)
MAX_LENGTH = 100
dataset = GibsonDataset(
    "test",
    seed=0,
    samples=10000,
    max_distance=MAX_LENGTH,
    episodes=20,
    ignore_0=False,
    debug=False,
    give_distance=True,
)
test_dataloader = data.DataLoader(
    dataset,
    batch_size=20,
    shuffle=True,
    num_workers=10,
)
mse = nn.MSELoss()
device = torch.device("cuda")
test_envs = np.load("./best_model/test_env.npy")
sparsifier = SiameseDeepVO().to(device).float()
sparsifier.device = device
sparsifier.load_state_dict(torch.load("./best_model/saved_model.pth"))
sparsifier.eval()
results = np.zeros((MAX_LENGTH + 2, 2))
results_pose_T = {}
results_pose_R = {}
predict_T = []
predict_R = []
data_T = []
data_R = []
for i in range(MAX_LENGTH + 2):
    results_pose_T[i] = []
    results_pose_R[i] = []
for i, batch in enumerate(tqdm(test_dataloader)):
    (x, y, d, poseGT) = batch
    d[torch.where(d >= MAX_LENGTH + 1)] = MAX_LENGTH
    d[torch.where(d == -1)] = MAX_LENGTH + 1
    poseGT = poseGT.to(device).float()
    hidden = sparsifier.init_hidden(y.shape[0], sparsifier.hidden_size, device)
    sparsifier.hidden = hidden
    image1, image2 = x
    image1 = image1.to(device).float()
    image2 = image2.to(device).float()
    pu.db
    pose, similarity = sparsifier(image1, image2, poseGT)
    # Handle distance estimation stuff
    for di, pGT, p in zip(d, poseGT, pose):
        T_GT = torch.norm(pGT[0:3])
        results_pose_T[di.item()].append(
            torch.sqrt(mse(p[0], T_GT)).detach().cpu().item()
        )
        results_pose_R[di.item()].append(
            torch.sqrt(mse(p[1], pGT[3])).detach().cpu().item()
        )
        predict_T.append(p[0].cpu().detach().numpy().item())
        predict_R.append(p[1].cpu().detach().numpy().item())
        data_T.append(T_GT.cpu().detach().numpy().item())
        data_R.append(pGT[3].cpu().detach().numpy().item())
    # Handle classifcaiton stuff
    result = torch.argmax(similarity, axis=1).detach().cpu()
    for di, r in zip(d, result):
        results[di][r] += 1
plt.hist(predict_T, color="r")
plt.title("Predictions of T")
plt.savefig("./results/predictionT_hist.png")
print("Max T = " + str(max(predict_T)) + " Min T = " + str(min(predict_T)))
print("Max R = " + str(max(predict_R)) + " Min R = " + str(min(predict_R)))
plt.clf()
plt.title("Predictions of R")
plt.hist(predict_R, color="r")
plt.savefig("./results/predictionR_hist.png")
plt.clf()
plt.title("Distribution of R")
plt.hist(data_R, color="r")
plt.savefig("./results/DistributionR.png")
plt.clf()
plt.title("Distribution of T")
plt.hist(data_T, color="r")
plt.savefig("./results/DistributionT.png")
plt.clf()
np.save("results.npy", results)
np.save("resultsPoseR.npy", results_pose_R, allow_pickle=True)
np.save("resultsPoseT.npy", results_pose_T, allow_pickle=True)
results = np.load("./results.npy")
results_pose_T = np.load("./resultsPoseT.npy", allow_pickle=True).item()
results_pose_R = np.load("./resultsPoseR.npy", allow_pickle=True).item()
ys = []
true = 0
total = 0
for i, row in enumerate(results):
    total += np.sum(row)
    if i >= 50:
        true += row[0]
    else:
        true += row[1]
print(true / total)
# Normalize by row
for counter, row in enumerate(results):
    s = np.sum(row)
    if s > 0:
        results[counter] = row / s
# ax = sns.heatmap(results, linewidths=0.1)
r = np.hstack([results[0:50, 1], results[50:, 0]])
plt.plot(r)

fig = matplotlib.pyplot.gcf()
fig.set_size_inches(18.5, 10.5)
plt.title("Model Similiarty Accuracy vs Distance")
plt.xlabel("Distance")
plt.ylabel("Accuracy")
plt.savefig("./results/plotSim.png", bbox_inches="tight", dpi=600)
results_T = np.zeros((MAX_LENGTH + 2, 1))
results_R = np.zeros((MAX_LENGTH + 2, 1))
plt.clf()
for row, key in results_pose_T.items():
    if len(key) > 0:
        results_T[row] = np.mean(key)
    else:
        results_T[row] = -1
for row, key in results_pose_R.items():
    if len(key) > 0:
        results_R[row] = np.mean(key)
    else:
        results_R[row] = -1
plt.title("Average RMSE Pose error vs Distance")
plt.plot(results_T, label="Translation")
plt.plot(results_R, label="Rotation")
plt.ylim(0, 2)
plt.xlim(0, 50)
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(18.5, 10.5)
plt.legend()

# fig = ax.get_figure()
plt.savefig("./results/plotPose.png", bbox_inches="tight", dpi=600)
