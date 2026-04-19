import numpy as np
import torch

# 🔥 FLATTEN weights (for secure aggregation)
def get_weights(model):
    flat = []
    for param in model.parameters():
        flat.extend(param.detach().numpy().flatten())
    return flat


# 🔥 RESTORE weights (VERY IMPORTANT)
def set_weights(model, flat_weights):
    pointer = 0

    for param in model.parameters():
        shape = param.data.shape
        size = np.prod(shape)

        new_vals = flat_weights[pointer:pointer+size]
        new_vals = np.array(new_vals).reshape(shape)

        param.data = torch.tensor(new_vals, dtype=torch.float32)

        pointer += size


# 🔥 FEDERATED AVERAGE (for flattened weights)
def federated_average(weights_list):
    return [
        sum(w[i] for w in weights_list) / len(weights_list)
        for i in range(len(weights_list[0]))
    ]