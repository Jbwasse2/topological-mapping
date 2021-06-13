import matplotlib.pyplot as plt
import pudb
import argparse
import numpy as np
import seaborn as sns
from matplotlib import rc
from matplotlib.ticker import MaxNLocator


rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
rc("text", usetex=True)
sns.set(style="darkgrid", font_scale=1.9)


def get_accuracy_plot(d):
    train_split = np.load(d + "/accuracy.npy")
    test_split = np.load(d + "/accuracy_v.npy")
    plt.plot(train_split, label="Training")
    plt.plot(test_split, label="Testing")
    plt.xlabel("Epochs")
    plt.ylabel("\% Accuracy")
    plt.xticks(range(0, len(train_split), 2))
    plt.title("Model Classification Accuracy vs Epochs")
    plt.legend()
    plt.savefig("./similarity_accuracy.pdf", bbox_inches="tight", dpi=600)
    plt.clf()


def get_losses_plot(d):
    train_split = np.load(d + "/losses.npy")
    test_split = np.load(d + "/losses_v.npy")
    plt.plot(train_split, label="Training")
    plt.plot(test_split, label="Testing")
    plt.xlabel("Epochs")
    plt.ylabel("Cross Entopy Loss")
    plt.xticks(range(0, len(train_split), 2))
    plt.title("Model Loss vs Epochs")
    plt.legend()
    plt.savefig("./similarity_losses.pdf", bbox_inches="tight", dpi=600)
    plt.clf()


parser = argparse.ArgumentParser("similarity_visualization")
parser.add_argument("dir", help="directory to Siamese pkl", type=str)
args = parser.parse_args()
get_accuracy_plot(args.dir)
get_losses_plot(args.dir)
