import numpy as np
import matplotlib.pyplot as plt

dist = np.load("disp.npy")
angles = np.load("angles.npy")

plt.hist(dist, bins=1000)
plt.show()
plt.clf()

plt.hist(angles, bins=1000)
plt.show()
plt.clf()
