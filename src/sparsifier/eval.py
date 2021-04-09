from data_getter import GibsonDataset
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import torch
import numpy as np
import pudb
from tqdm import tqdm
from best_model.model import Siamese

sns.color_palette("flare", as_cmap=True)

dataset = GibsonDataset(
    "test",
    seed=0,
    samples=200,
    max_distance=30,
    episodes=20,
    ignore_0=False,
    debug=False,
    give_distance=True,
)
device = torch.device("cuda:0")
test_envs = np.load("./best_model/test_env.npy")
sparsifier = Siamese().to(device)
sparsifier.load_state_dict(torch.load("./best_model/saved_model.pth"))
sparsifier.eval()
results = np.zeros((31, 2))
for batch in tqdm(dataset):
    (x, y, d) = batch
    if d >= 31:
        d = 30
    image1, image2 = x
    image1 = image1.to(device).float().unsqueeze(0)
    image2 = image2.to(device).float().unsqueeze(0)
    result = F.softmax(sparsifier(image1, image2))
    result = torch.argmax(result)
    results[d][result] += 1
np.save("results.npy", results)
results = np.load("./results.npy")
pu.db
ax = sns.heatmap(results, linewidths=0.1, annot=True)
true = 0
total = 0
for i, row in enumerate(results):
    total += np.sum(row)
    if i >= 15:
        true += row[0]
    else:
        true += row[1]
print(true / total)

fig = ax.get_figure()
fig.savefig("plot.png")
