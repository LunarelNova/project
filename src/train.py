import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .federated import get_weights


def train_local(model, X, y, epochs=5):
    """
    Train a model locally on a client's dataset.

    Returns:
        weights: list of numpy arrays (same shape as model params)
        loss: final training loss
    """

    model.train()

    # =========================
    # 🔥 FIX 1: HANDLE PANDAS / NUMPY SAFELY
    # =========================
    if hasattr(X, "values"):
        X = X.values
    if hasattr(y, "values"):
        y = y.values

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # Ensure proper shapes
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)

    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    # =========================
    # 🔥 FIX 2: CONVERT TO TENSORS
    # =========================
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # =========================
    # 🔥 FIX 3: LOSS + OPTIMIZER
    # =========================
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # =========================
    # 🔁 TRAIN LOOP
    # =========================
    for epoch in range(epochs):
        optimizer.zero_grad()

        outputs = model(X_tensor)

        # 🔥 IMPORTANT: ensure output shape matches y
        if outputs.shape != y_tensor.shape:
            outputs = outputs.view_as(y_tensor)

        loss = criterion(outputs, y_tensor)

        loss.backward()
        optimizer.step()

    # =========================
    # 🔥 FIX 4: RETURN STRUCTURED WEIGHTS (NOT FLATTENED)
    # =========================
    weights = get_weights(model)

    return weights, float(loss.item())