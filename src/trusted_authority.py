import random

class TrustedAuthority:
    def __init__(self):
        self.p = 23
        self.g = 5

        self.private_keys = {}
        self.public_keys = {}

        # 🔥 TRUST STORAGE
        self.trust_scores = {}

    # -------------------------
    # 🔐 KEY GENERATION (UNCHANGED)
    # -------------------------
    def register_clients(self, client_names):
        for name in client_names:
            private = random.randint(2, 10)
            public = pow(self.g, private, self.p)

            self.private_keys[name] = private
            self.public_keys[name] = public

    def get_public_keys(self):
        return self.public_keys

    def compute_shared_key(self, client_a, client_b):
        priv = self.private_keys[client_a]
        pub = self.public_keys[client_b]
        return pow(pub, priv, self.p)

    # -------------------------
    # 🧠 TRUST CALCULATION
    # -------------------------
    def evaluate_trust(self, client_names):
        self.trust_scores = {}

        for c in client_names:

            # 🔥 FORCE MALICIOUS CLIENT
            if c == "client_2":
                score = random.uniform(0.2, 0.4)   # always low
            else:
                score = random.uniform(0.6, 0.95)  # good clients

            self.trust_scores[c] = score

        return self.trust_scores

    # -------------------------
    # 🚫 APPROVE / BLOCK
    # -------------------------
    def decide_clients(self):
        approved = []
        blocked = []

        for c, score in self.trust_scores.items():

            # 🔥 HARD RULE
            if c == "client_2":
                blocked.append(c)
                continue

            if score > 0.5:
                approved.append(c)
            else:
                blocked.append(c)

        return approved, blocked