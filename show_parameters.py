import torch
import numpy as np
from model import CNN

model = CNN()
params = np.load("reports/params_15clients/global_round_5.npz")

with torch.no_grad():
    for p, arr in zip(model.parameters(), params.values()):
        p.copy_(torch.tensor(arr))

print("First layer weights (sample):")
print(model.conv1.weight[0, 0])
