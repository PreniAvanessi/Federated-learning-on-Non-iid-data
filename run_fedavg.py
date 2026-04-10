# run_fedavg.py
import os
import time
import random

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import flwr as fl
import numpy as np
import torch
from flwr.common import Context, Metrics, ndarrays_to_parameters
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from client import CNNClient
from noniid_partition import dirichlet_noniid_partition, quantity_noniid_partition
from model import CNN


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Distributed metrics aggregation (client-side evaluate metrics)
# -----------------------------
def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    accuracies = [m.get("accuracy", 0.0) for _, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    if not accuracies or sum(examples) == 0:
        return {"accuracy": 0.0, "min_accuracy": 0.0, "accuracy_variance": 0.0}

    weighted_acc = np.average(accuracies, weights=examples)
    worst_acc = float(min(accuracies))
    acc_variance = float(np.var(accuracies))

    return {
        "accuracy": float(weighted_acc),
        "min_accuracy": worst_acc,
        "accuracy_variance": acc_variance,
    }


# -----------------------------
# Dataset Loader
# -----------------------------
def load_dataset(dataset_name: str):
    if dataset_name == "cifar10":
        stats = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])
        train_data = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10("./data", train=False, download=True, transform=transform)
        channels = 3

    elif dataset_name == "mnist":
        stats = ((0.1307,), (0.3081,))
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])
        train_data = datasets.MNIST("./data", train=True, download=True, transform=transform)
        test_data = datasets.MNIST("./data", train=False, download=True, transform=transform)
        channels = 1

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train_data, test_data, channels


# -----------------------------
# Centralized Evaluation (server-side)
# -----------------------------
def get_evaluate_fn(dataset_name: str):
    _, test_data, channels = load_dataset(dataset_name)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

    def evaluate(server_round, parameters, config):
        model = CNN(input_channels=channels)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        criterion = torch.nn.CrossEntropyLoss()
        loss_sum, correct, total = 0.0, 0, 0
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                loss = criterion(outputs, labels)
                loss_sum += loss.item() * labels.size(0)

                preds = outputs.argmax(dim=1)
                probs = torch.softmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                total += labels.size(0)
                correct += (preds == labels).sum().item()

        avg_loss = loss_sum / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        f1 = f1_score(all_labels, all_preds, average="macro") if total > 0 else 0.0

        try:
            auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")
        except ValueError:
            auc = 0.0

        return avg_loss, {"accuracy": float(accuracy), "f1_score": float(f1), "auc": float(auc)}

    return evaluate


# -----------------------------
# Main Experiment
# -----------------------------
def run_experiment(
    dataset_name: str,
    num_clients: int,
    num_rounds: int,
    seed: int = 42,
    skew_type: str = "label",
    alpha: float = 0.3,
    local_epochs: int = 1,
):
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # short GPU/CPU confirmation
    if device.type == "cuda":
        print(f"Device=cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("Device=cpu")

    print(
        f"FedAvg | {dataset_name} | K={num_clients} | R={num_rounds} | "
        f"skew={skew_type} | alpha={alpha} | E={local_epochs} | seed={seed}"
    )

    full_train, full_test, channels = load_dataset(dataset_name)

    # Non-IID partition
    if skew_type == "label":
        client_indices = dirichlet_noniid_partition(
            full_train, num_clients=num_clients, alpha=alpha, num_classes=10, seed=seed
        )
    elif skew_type == "quantity":
        client_indices = quantity_noniid_partition(full_train, num_clients=num_clients, alpha=alpha, seed=seed)
    else:
        raise ValueError("skew_type must be 'label' or 'quantity'")

    def fit_config(server_round: int):
        return {"local_epochs": int(local_epochs)}

    def client_fn(context: Context):
        try:
            cid = int(context.node_id)
        except (ValueError, TypeError):
            cid = 0
        cid = cid % num_clients

        train_subset = Subset(full_train, client_indices[cid])

        g = torch.Generator()
        g.manual_seed(seed + cid)

        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, generator=g)
        test_loader = DataLoader(full_test, batch_size=256, shuffle=False)

        return CNNClient(
            train_loader,
            test_loader,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            num_channels=channels,
        ).to_client()

    # traffic estimate
    dummy_model = CNN(input_channels=channels)
    param_count = sum(p.numel() for p in dummy_model.parameters())
    model_size_bytes = param_count * 4
    round_traffic_bytes = model_size_bytes * num_clients * 2  # download + upload

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_metrics_aggregation_fn=weighted_average,  # ✅ gives min_accuracy in distributed metrics
        evaluate_fn=get_evaluate_fn(dataset_name),   #server-side evaluation on the global test set.
        initial_parameters=ndarrays_to_parameters([p.detach().cpu().numpy() for p in dummy_model.parameters()]),  #sets the starting global model weights
        on_fit_config_fn=fit_config,  #Sends config to clients each round
    )

    start_time = time.time()
    hist = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        ray_init_args={"num_cpus": 1, "include_dashboard": False},
    )
    total_runtime = time.time() - start_time

    # centralized metrics
    cent_acc = [v for _, v in hist.metrics_centralized.get("accuracy", [])]
    cent_f1 = [v for _, v in hist.metrics_centralized.get("f1_score", [])]
    cent_auc = [v for _, v in hist.metrics_centralized.get("auc", [])]

    # distributed metrics (from client-side evaluation aggregation)
    dist_min_acc = [v for _, v in hist.metrics_distributed.get("min_accuracy", [])]

    losses = [v for _, v in hist.losses_centralized]

    rows = []
    n = min(len(losses), num_rounds)  # avoids the "extra row" issue if Flower returns round 0
    for i in range(n):
        round_num = i + 1
        rows.append({
            "round": round_num,
            "global_loss": float(losses[i]),
            "global_accuracy": float(cent_acc[i]) if i < len(cent_acc) else 0.0,
            "f1_score": float(cent_f1[i]) if i < len(cent_f1) else 0.0,
            "auc": float(cent_auc[i]) if i < len(cent_auc) else 0.0,
            "worst_client_acc": float(dist_min_acc[i]) if i < len(dist_min_acc) else 0.0,
            "round_traffic_bytes": int(round_traffic_bytes),
            "total_traffic_bytes": int(round_traffic_bytes * round_num),
            "total_runtime": float(total_runtime),
            "num_clients": int(num_clients),
            "dataset": str(dataset_name),
            "skew_type": str(skew_type),
            "alpha": float(alpha),
            "local_epochs": int(local_epochs),
            "seed": int(seed),
        })

    return rows


if __name__ == "__main__":
    run_experiment(
        dataset_name="cifar10",
        num_clients=10,
        num_rounds=20,
        seed=42,
        skew_type="label",
        alpha=0.3,
        local_epochs=5,
    )