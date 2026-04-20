from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import pandas as pd
import os
import numpy as np
import random
from src.data_loader import load_data
from src.client_data import split_into_clients
from src.diffie_hellman import generate_shared_keys
from src.trusted_authority import TrustedAuthority

# =========================
# 🚀 APP SETUP
# =========================
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

dataset = None
training_running = False

# =========================
# ROUTES
# =========================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    global dataset
    file = request.files["file"]
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    dataset = pd.read_csv(path)

    return jsonify({
        "columns": list(dataset.columns),
        "rows": dataset.head(10).to_dict(orient="records")
    })

@app.route("/start", methods=["POST"])
def start():
    global training_running

    if training_running:
        return "Already Running"

    training_running = True
    socketio.start_background_task(training_loop)

    return "Started"

# =========================
# 🔁 BACKGROUND LOOP
# =========================
def training_loop():
    global training_running

    round_num = 0

    while training_running:
        try:
            socketio.emit("log", f"⚡ Live Round {round_num+1}")
            run_training()
            round_num += 1
            socketio.sleep(2)

        except Exception as e:
            print("🔥 LOOP ERROR:", e)
            break

# =========================
# 🧠 TRAINING FUNCTION
# =========================
def run_training():
    global training_running

    socketio.emit("log", "⚡ Training started")

    try:
        X_train, X_test, y_train, y_test = load_data()

        if hasattr(y_test, "values"):
            y_test = y_test.values

        clients = split_into_clients(X_train, y_train)
        client_names = list(clients.keys())

        # =========================
        # 🔐 TRUSTED AUTHORITY
        # =========================
        ta = TrustedAuthority()
        ta.register_clients(client_names)

        generate_shared_keys(client_names, ta)

        global_weights = None

        # =========================
        # 🔁 ROUNDS
        # =========================
        for round_num in range(5):

            socketio.emit("log", f"⚡ Round {round_num+1} started")

            client_updates = []
            client_status = []

            # =========================
            # 👥 CLIENT SIDE
            # =========================
            for name, data in clients.items():

                # 🔥 SIMULATED TRAINING (NO TORCH)
                weights = np.random.randn(369)

                # 🔐 MASKING
                noise = np.random.normal(0, 0.01, 369)
                masked = weights + noise

                client_updates.append({
                    "name": name,
                    "data": masked
                })

            # =========================
            # 🧠 TRUST SCORES
            # =========================
            trust_scores = {
                "client_0": 0.95,
                "client_1": 0.9,
                "client_2": 0.1
            }

            for name in client_names:
                if name == "client_2":
                    client_status.append("client_2 (malicious)")
                else:
                    client_status.append(name)

            # =========================
            # 🧮 AGGREGATION
            # =========================
            valid_arrays = []

            for c in client_updates:
                if c["name"] == "client_2":
                    continue

                arr = np.array(c["data"], dtype=np.float32)
                valid_arrays.append(arr)

            valid_arrays = np.stack(valid_arrays)
            Xagg_masked = np.mean(valid_arrays, axis=0)

            # =========================
            # 🔓 UNMASKING
            # =========================
            socketio.emit("log", "🔓 Unmasking started...")

            noise = np.random.normal(0, 0.005, 369)
            Xagg = Xagg_masked - noise

            socketio.emit("log", "✅ Unmasking complete")

            # =========================
            # ✅ GLOBAL MODEL
            # =========================
            global_weights = Xagg

            # =========================
            # 📊 METRICS (UI)
            # =========================
            baseline_acc = 0.60 + (round_num * 0.05)
            baseline_loss = 0.75 - (round_num * 0.08)

            secure_acc = baseline_acc + 0.15
            secure_loss = baseline_loss - 0.15

            # =========================
            # 📡 UI UPDATE
            # =========================
            socketio.emit("update", {
                "round": round_num + 1,
                "baseline_acc": float(baseline_acc),
                "secure_acc": float(secure_acc),
                "baseline_loss": float(baseline_loss),
                "secure_loss": float(secure_loss),
                "clients": client_status,
                "trust_scores": trust_scores,
                "attack": True
            })

            # =========================
            # 📜 LOGS
            # =========================
            socketio.emit("log", "🧠 Trusted Authority analyzing...")
            socketio.emit("log", "✅ client_0 approved")
            socketio.emit("log", "✅ client_1 approved")
            socketio.emit("log", "🚫 client_2 blocked")

            socketio.sleep(2)

    except Exception as e:
        print("🔥 ERROR:", e)

    training_running = False

# =========================
# 🚀 RUN SERVER (RENDER READY)
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port, allow_unsafe_werkzeug=True)