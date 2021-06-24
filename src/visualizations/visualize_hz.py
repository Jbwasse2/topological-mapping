from matplotlib import rc
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import pudb
from matplotlib.pyplot import figure
import re
import numpy as np

figure(figsize=(20, 12), dpi=700)

rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
rc("text", usetex=True)
sns.set(style="darkgrid", font_scale=1.9)


def get_data(d):
    hz = []
    f = open(d)
    time_to_run = []
    for line in f:
        numbers = re.findall(r"\d+.\d+", line)
        hz.append(1 / float(numbers[-1]))
    return hz


hz1 = get_data("./data/runlogEKF")
plt.title("Update Rate of Topological Map (hz)")
plt.ylabel("hz")
plt.xlabel("frame update \#")
plt.ylim(0, 200)
plt.plot(hz1, label="Using EKF")
hz2 = get_data("./data/runlogNoEKF")
plt.plot(hz2, label="Not Using EKF")
plt.plot([np.mean(hz1) for _ in range(len(hz1))], label="Average hz Using EKF")
plt.plot([np.mean(hz2) for _ in range(len(hz2))], label="Average hz Without EKF")
plt.legend()
plt.savefig("./ekf_hz.pdf", dpi=700, bbox_inches="tight")
