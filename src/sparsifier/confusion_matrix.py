# this takes a trained network as input and creates a
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from data.results.sparsifier.best_model.model import Siamese
from tqdm import tqdm
from data_getter import GibsonDataset
from torch.utils import data
import os
import pudb
import numpy as np
import argparse

import torch

matplotlib.use("Agg")

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

seed = 0
device = torch.device("cuda:0")
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--directory_location", help="No", required=True)
args = parser.parse_args()

model_path = os.path.join(args.directory_location, "saved_model.pth")
model = Siamese().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()
NUMBER_OF_SAMPLES = 10000
test_dataset = GibsonDataset("test", seed, samples=NUMBER_OF_SAMPLES)
test_dataloader = data.DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=16,
)

confusion = np.zeros((11, 11))
for i, batch in enumerate(tqdm(test_dataloader)):
    x, y = batch
    im1, im2 = x
    y = y.to(device)
    im1 = im1.to(device).float()
    im2 = im2.to(device).float()
    out = model(im1, im2)
    choose = np.argmax(out.cpu().detach().numpy(), axis=1)
    y = y.cpu().detach().numpy()
    for truth, prediction in zip(y, choose):
        confusion[truth, prediction] += 1
confusion = confusion / NUMBER_OF_SAMPLES
a4_dims = (16, 16)
fig, ax = plt.subplots(figsize=a4_dims)
sns.heatmap(confusion, annot=True, fmt=".1%")
plt.savefig("confusion.jpg", bbox_inches="tight")
