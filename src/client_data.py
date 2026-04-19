import numpy as np
from .config import NUM_CLIENTS

def split_into_clients(X, y):
    data = list(zip(X, y))  # 🔥 safer
    np.random.shuffle(data)

    split_size = len(data) // NUM_CLIENTS
    clients = {}

    for i in range(NUM_CLIENTS):
        subset = data[i * split_size:(i + 1) * split_size]

        X_c, y_c = zip(*subset)

        clients[f"client_{i}"] = {
            "X": np.array(X_c),
            "y": np.array(y_c)
        }

    return clients