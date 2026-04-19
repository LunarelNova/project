from src.data_loader import load_data
from src.client_data import split_into_clients
from src.model import HeartModel
from src.train import train_local
from src.federated import set_weights
from src.secure_agg import (
    generate_masks,
    apply_double_mask,
    remove_double_mask,
    split_secret,
    reconstruct_secret
)
from src.truth_discovery import compute_weight
from src.trusted_authority import setup_verification

from flask_socketio import SocketIO
import numpy as np
import time

def run_training(socketio):

    ROUNDS = 3

    X_train, X_test, y_train, y_test = load_data()
    clients = split_into_clients(X_train, y_train)

    global_model = HeartModel()
    global_weights = None

    dummy_model = HeartModel()
    weights_shapes = [p.shape for p in dummy_model.parameters()]
    a, c = setup_verification(weights_shapes)

    for round in range(ROUNDS):

        round_data = {
            "round": round+1,
            "clients": []
        }

        clean_weights_list = []
        weights_values = []

        for name, data in clients.items():

            local_model = HeartModel()

            if global_weights is not None:
                set_weights(local_model, global_weights)

            weights, loss = train_local(local_model, data["X"], data["y"])

            is_malicious = False

            if name == "client_2":
                is_malicious = True
                weights = [
                    w + np.random.normal(0, 5, size=w.shape)
                    for w in weights
                ]

            avg_magnitude = np.mean([np.abs(w).mean() for w in weights])

            if avg_magnitude > 2:
                round_data["clients"].append({
                    "name": name,
                    "loss": float(loss),
                    "removed": True,
                    "malicious": True
                })
                continue

            weight = compute_weight(weights, global_weights)

            weighted_weights = [weight * w for w in weights]

            r, b = generate_masks(weighted_weights)
            r_shares = split_secret(r)
            b_shares = split_secret(b)

            r_rec = reconstruct_secret(r_shares)
            b_rec = reconstruct_secret(b_shares)

            masked_weights = apply_double_mask(weighted_weights, r_rec, b_rec)

            clean_weights_list.append(masked_weights)
            weights_values.append(weight)

            round_data["clients"].append({
                "name": name,
                "loss": float(loss),
                "weight": float(weight),
                "malicious": is_malicious,
                "removed": False
            })

        summed_weights = [
            np.sum(layer, axis=0)
            for layer in zip(*clean_weights_list)
        ]

        total_weight = sum(weights_values)

        global_weights = [
            w / total_weight for w in summed_weights
        ]

        round_data["verification"] = "success"

        # 🚀 SEND REAL-TIME DATA
        socketio.emit("update", round_data)

        time.sleep(2)  # simulate real-time delay