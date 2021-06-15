# Used to visualize pkls created from orbslam
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pudb

name = "./gibson/Browntown.pkl"
fp = open(name, "rb")
b = pickle.load(fp)
ids = [int(i[0]) for i in b]
ids_zerod = [i - min(ids) for i in ids]
x = range(len(ids_zerod))
# Plot 1 or 0 depending on if is inf
ids2 = [i[1] for i in b]
labels2 = [False if i == np.inf else True for i in ids2]
plt.plot(x, labels2)
plt.show()
# Plot (x,y)
xy = np.array(b)[labels2].astype(np.float)  # Only get rows that are good
# Only plot first subset of localized points
labels3 = [True if int(i[0]) > 3000 else False for i in xy]
xy = xy[labels3]
x_pose = list(xy[:, 1])
y_pose = list(xy[:, 2])
plt.plot(x_pose, y_pose)
plt.axis([min(x_pose), max(x_pose), min(y_pose), max(y_pose)])
plt.title(
    name + " Trajectory Frame " + str(min(xy[:, 0])) + " to Frame " + str(max(xy[:, 0]))
)
plt.show()
