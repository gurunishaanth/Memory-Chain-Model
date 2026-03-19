import torch.nn as nn
import torch
import Memory_cell
# ---- Synthetic ununiform dataset (sparse patterns) ----
dataset = [
    {0:1, 1:1, 2:1},
    {0:1, 1:1, 3:1},
    {0:1, 1:1, 2:1},
    {5:1, 6:1},
    {5:1, 6:1, 7:1},
    {0:1, 1:1, 2:1},
    {20:1},          # anomaly-like
    {5:1, 6:1},
]

# ---- Run model ----
model = Memory_cell.MemoryCell(input_size=21, pattern_size=10, learning_rate=0.1, Sim_Thr=0.5, Ano_Thr=0.3)

results = []
for t, x in enumerate(dataset):
    k, conf, anomaly = model.step(x)
    results.append((t, x, k, round(conf, 2), anomaly))

results