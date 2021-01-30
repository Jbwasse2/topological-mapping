import numpy as np
import pudb
import seaborn as sns
import matplotlib.pyplot as plt

def get_data(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        info = f.readlines()[-10]
    return info

heatmap = np.zeros((20,20))
for i in range(1,20):
    for j in range(1,20):
        file_name = str(i) + "_" + str(j)
        try:
            data = eval(get_data(file_name))
            heatmap[i][j] = data[0]
        except Exception as e:
            pass
ax = sns.heatmap(heatmap, linewidths=0.2)
plt.show()

