import numpy as np

def compute_weights(client_updates):
    """
    Assign weights based on similarity to mean update
    """

    avg = np.mean(client_updates, axis=0)

    weights = []

    for update in client_updates:
        distance = np.linalg.norm(update - avg)

        # smaller distance = more trustworthy
        weight = 1 / (distance + 1e-5)

        weights.append(weight)

    # normalize
    weights = np.array(weights)
    weights = weights / np.sum(weights)

    return weights