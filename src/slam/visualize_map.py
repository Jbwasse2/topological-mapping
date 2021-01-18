import torch
import numpy as np
import pudb
import matplotlib.pyplot as plt

a = torch.load('./map2D.pt').cpu().detach().numpy().squeeze()
plt.imshow(a)
plt.show()
a = torch.load('./obstacles.pt').cpu().detach().numpy().squeeze()
plt.imshow(a)
plt.show()
