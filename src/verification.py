import numpy as np

def generate_tag(weights):
    tag = 0

    for w in weights:
        tag += np.sum(w)

    return float(tag)


def verify_aggregation(weights_list, tags):
    computed = []

    for weights in weights_list:
        computed.append(generate_tag(weights))

    # Compare tags
    for i in range(len(tags)):
        if abs(tags[i] - computed[i]) > 1e-3:
            return False

    return True