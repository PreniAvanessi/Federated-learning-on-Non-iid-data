import numpy as np

def iid_partition(dataset, num_clients):
    num_items = len(dataset)
    indices = np.random.permutation(num_items)
    
    split_size = num_items // num_clients
    client_indices = []

    for i in range(num_clients):
        start = i * split_size
        end = start + split_size
        client_indices.append(indices[start:end])

    return client_indices
