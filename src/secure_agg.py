shared_keys = {}

def set_shared_keys(keys):
    global shared_keys
    shared_keys = keys


def mask_weights(name, weights, client_names):
    masked = []

    for w in weights:
        total_mask = 0

        for other in client_names:
            if other == name:
                continue

            key = shared_keys.get((name, other), 0)

            # 🔐 pairwise mask (+ or - based on order)
            if name < other:
                total_mask += key * 0.01
            else:
                total_mask -= key * 0.01

        masked.append(w + total_mask)

    return masked


def aggregate_masked(masked_updates):
    agg = []

    for i in range(len(masked_updates[0])):
        agg.append(
            sum(client[i] for client in masked_updates) / len(masked_updates)
        )

    return agg


def unmask_aggregate(agg_weights, client_names):
    # masks cancel out automatically
    return agg_weights