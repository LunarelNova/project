import random

p = 23
g = 5

users = {}

def generate_keys(client_names):
    global users
    users = {}

    for name in client_names:
        private = random.randint(1, 10)
        public = pow(g, private, p)

        users[name] = {
            "private": private,
            "public": public
        }

    return {name: users[name]["public"] for name in users}


def generate_shared_keys(client_names, ta):
    shared_keys = {}

    for i in client_names:
        for j in client_names:
            if i != j:
                key = ta.compute_shared_key(i, j)
                shared_keys[(i, j)] = key

    return shared_keys