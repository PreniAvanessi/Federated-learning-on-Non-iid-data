# non_iid_partition.py
import numpy as np


def _get_labels(dataset):
    """
    Extract labels from common torchvision-style datasets.
    Falls back to indexing the dataset (slower).
    """
    if hasattr(dataset, "targets"):
        return np.array(dataset.targets)
    if hasattr(dataset, "labels"):
        return np.array(dataset.labels)
    # Fallback: dataset[i] -> (x, y)
    return np.array([dataset[i][1] for i in range(len(dataset))])


def dirichlet_noniid_partition(
    dataset,
    num_clients: int,
    alpha: float = 0.3,
    num_classes: int = 10,
    seed: int = 42,
    min_size_per_client: int = 10,
):
    """
    Dirichlet-based non-IID partitioning.

    For each class c, we distribute its samples across clients according to
    proportions drawn from Dirichlet(alpha).

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The dataset to split (e.g., CIFAR-10 train set).
    num_clients : int
        Number of federated clients.
    alpha : float
        Dirichlet concentration parameter. Smaller => more non-IID.
    num_classes : int
        Number of classes (CIFAR-10 = 10).
    seed : int
        Random seed for reproducibility.
    min_size_per_client : int
        Ensures each client has at least this many samples (best-effort).

    Returns
    -------
    list[np.ndarray]
        A list of index arrays, one per client.
    """
    if num_clients <= 0:
        raise ValueError("num_clients must be > 0")
    if alpha <= 0:
        raise ValueError("alpha must be > 0")

    rng = np.random.default_rng(seed)    #random number generstor with a fixed seed
    labels = _get_labels(dataset)

    # Collect indices for each class
    class_indices = [np.where(labels == c)[0] for c in range(num_classes)]  #It groups dataset indices by class and then randomly shuffles them so the data can be fairly distributed across clients
    for c in range(num_classes):
        rng.shuffle(class_indices[c])

    # We'll retry a few times to avoid empty/tiny clients (common when alpha is small)
    max_retries = 30

    for _ in range(max_retries):
        client_indices = [[] for _ in range(num_clients)]

        for c in range(num_classes):
            idx_c = class_indices[c]
            if len(idx_c) == 0:
                continue

            # Sample proportions for this class across clients
            proportions = rng.dirichlet(np.full(num_clients, alpha))

            # Turn proportions into counts (handle rounding carefully)
            counts = np.floor(proportions * len(idx_c)).astype(int)

            # Fix rounding mismatch so sum(counts) == len(idx_c)
            diff = len(idx_c) - counts.sum()
            if diff > 0:
                # add remaining samples to clients with largest fractional needs (randomized)
                for i in rng.choice(num_clients, size=diff, replace=True):
                    counts[i] += 1
            elif diff < 0:
                # remove extras from clients with count>0
                removable = np.where(counts > 0)[0]
                for i in rng.choice(removable, size=abs(diff), replace=True):
                    counts[i] -= 1

            # Assign indices to clients
            start = 0
            for client_id, ct in enumerate(counts):
                end = start + ct
                if ct > 0:
                    client_indices[client_id].extend(idx_c[start:end].tolist())
                start = end

        sizes = np.array([len(idxs) for idxs in client_indices])
        if sizes.min() >= min_size_per_client:
            # Shuffle each client's indices and return
            return [rng.permutation(np.array(idxs, dtype=int)) for idxs in client_indices]

    # If we can't satisfy min_size_per_client, return the last attempt anyway (still valid split)
    return [rng.permutation(np.array(idxs, dtype=int)) for idxs in client_indices]     #permutation=shuffling


def summarize_client_labels(dataset, client_indices, num_classes: int = 10):
    """
    Utility: prints per-client label distribution summary for debugging/reporting.
    """
    labels = _get_labels(dataset)
    for i, idxs in enumerate(client_indices):
        counts = np.bincount(labels[idxs], minlength=num_classes)
        top = counts.argsort()[::-1][:3]
        top_str = ", ".join([f"{t}:{int(counts[t])}" for t in top if counts[t] > 0])
        print(f"Client {i:02d} | n={len(idxs):5d} | top labels: {top_str}")



def quantity_noniid_partition(
    dataset,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
    min_size: int = 50,
):
    """
    Quantity-skew non-IID partitioning.
    
    Clients receive different numbers of samples, but labels are approximately IID.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset to split.
    num_clients : int
        Number of federated clients.
    alpha : float
        Dirichlet concentration parameter controlling imbalance.
        Smaller => more skew.
    seed : int
        Random seed.
    min_size : int
        Minimum number of samples per client.

    Returns
    -------
    list[np.ndarray]
        List of index arrays, one per client.
    """
    rng = np.random.default_rng(seed)

    num_samples = len(dataset)
    indices = rng.permutation(num_samples)

    # Sample client proportions (quantity skew)
    proportions = rng.dirichlet(np.full(num_clients, alpha))

    # Convert proportions to counts
    counts = (proportions * num_samples).astype(int)

    # Fix rounding issues
    diff = num_samples - counts.sum()
    for i in rng.choice(num_clients, size=abs(diff), replace=True):
        counts[i] += 1 if diff > 0 else -1

    # Enforce minimum size per client
    for i in range(num_clients):
        if counts[i] < min_size:
            deficit = min_size - counts[i]
            donor = np.argmax(counts)
            counts[donor] -= deficit
            counts[i] += deficit

    # Assign indices
    client_indices = []
    start = 0
    for c in counts:
        end = start + c
        client_indices.append(indices[start:end])
        start = end

    return client_indices