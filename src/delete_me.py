import torch
from map.worm_model.model import Siamese

device = torch.device("cuda:0")
model = Siamese().to(device)
model.load_state_dict(torch.load("./map/worm_model/saved_model.pth"))
model.eval()
import time

N = 600

start_time = time.time()
for i in range(N):
    a = torch.rand(1, 3, 224, 224).to(device)
    b = torch.rand(1, 3, 224, 224).to(device)
    model(a, b)
print(time.time() - start_time)

start_time = time.time()
a = torch.rand(N, 3, 224, 224).to(device)
b = torch.rand(N, 3, 224, 224).to(device)
model(a, b)
print(time.time() - start_time)
