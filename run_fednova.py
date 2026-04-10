import os
import shutil
import time

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import flwr as fl
import numpy as np
import torch
from flwr.common import Context, Metrics, ndarrays_to_parameters, parameters_to_ndarrays
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from client import CNNClient
from noniid_partition import dirichlet_noniid_partition, quantity_noniid_partition
from model import CNN
import random

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    accuracies = [m.get("accuracy", 0.0) for _, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    if not accuracies or sum(examples) == 0:
        return {
            "accuracy": 0.0,
            "min_accuracy": 0.0,
            "accuracy_variance": 0.0,
        }

    weighted_acc = np.average(accuracies, weights=examples)
    worst_acc = min(accuracies)
    acc_variance = float(np.var(accuracies))   # ✅ THIS IS NEW

    return {
        "accuracy": float(weighted_acc),
        "min_accuracy": float(worst_acc),
        "accuracy_variance": acc_variance,       # ✅ RETURNED TO FLOWER
    }



def load_dataset(dataset_name: str):
    if dataset_name == "mnist":
        stats = ((0.1307,), (0.3081,))
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])
        train_data = datasets.MNIST("./data", train=True, download=True, transform=transform)
        test_data = datasets.MNIST("./data", train=False, download=True, transform=transform)
        channels = 1

    elif dataset_name == "fashion_mnist":
        stats = ((0.2860,), (0.3530,))
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])
        train_data = datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
        test_data = datasets.FashionMNIST("./data", train=False, download=True, transform=transform)
        channels = 1

    elif dataset_name == "cifar10":
        stats = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])
        train_data = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10("./data", train=False, download=True, transform=transform)
        channels = 3

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train_data, test_data, channels


def get_evaluate_fn(dataset_name: str):
    _, test_data, channels = load_dataset(dataset_name)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: dict):
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

                _, predicted = torch.max(outputs.data, 1)
                probs = torch.softmax(outputs, dim=1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = loss_sum / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        f1 = f1_score(all_labels, all_preds, average="macro") if total > 0 else 0.0

        try:
            auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")
        except ValueError:
            auc = 0.0

        return avg_loss, {"accuracy": accuracy, "f1_score": f1, "auc": auc}

    return evaluate


class SaveFedNovaStrategy(fl.server.strategy.FedAvg):
    def __init__(self, save_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

        # Keep a copy of current global params (w_t)
        if self.initial_parameters is None:
            raise ValueError("initial_parameters must be set for FedNovaStrategy")
        self.current_global = parameters_to_ndarrays(self.initial_parameters)

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        w_t = self.current_global  # global params BEFORE aggregation

        # total examples for p_k weights
        total_examples = sum(fitres.num_examples for _, fitres in results)
        if total_examples == 0:
            return None, {}

        deltas_hat = []   # normalized updates (Δ_k / τ_k)
        pks = []          # p_k = n_k / n_total
        taus = []         # τ_k = num_steps

        for _, fitres in results:
            w_k = parameters_to_ndarrays(fitres.parameters)

            tau_k = int(fitres.metrics.get("num_steps", 1))
            tau_k = max(tau_k, 1)

            # Δ_k = w_k - w_t
            delta_k = [wk - wt for wk, wt in zip(w_k, w_t)]

            # Δ̂_k = Δ_k / τ_k
            delta_hat_k = [d / tau_k for d in delta_k]

            deltas_hat.append(delta_hat_k)
            pks.append(fitres.num_examples / total_examples)
            taus.append(tau_k)

        pks = np.array(pks, dtype=np.float64)
        taus = np.array(taus, dtype=np.float64)

        # τ_eff = Σ p_k τ_k  (important scaling)
        tau_eff = float(np.sum(pks * taus))
        tau_eff = max(tau_eff, 1.0)

        # Aggregate normalized deltas: Σ p_k Δ̂_k
        agg_delta_hat = []
        for layer_i in range(len(w_t)):
            layer_stack = np.stack([d[layer_i] for d in deltas_hat], axis=0)  # (K, ...)
            agg_layer = np.tensordot(pks, layer_stack, axes=(0, 0))
            agg_delta_hat.append(agg_layer)

        # FedNova step: w_{t+1} = w_t + τ_eff * Σ p_k Δ̂_k
        w_next = [wt + tau_eff * d_hat for wt, d_hat in zip(w_t, agg_delta_hat)]

        self.current_global = w_next
        aggregated_parameters = ndarrays_to_parameters(w_next)

        # save
        #np.savez(os.path.join(self.save_path, f"round-{server_round}-weights.npz"), *w_next)

        return aggregated_parameters, {}




def run_experiment(
    dataset_name: str,
    num_clients: int,
    num_rounds: int,
    seed: int = 42,
    skew_type: str = "label",  # "label" or "quantity"
    alpha: float = 0.3,
    local_epochs: int = 1,     # ✅ NEW
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print(
        f"FedNova(style) | dataset={dataset_name} clients={num_clients} rounds={num_rounds} "
        f"skew_type={skew_type} alpha={alpha} local_epochs={local_epochs}"
    )

    os.makedirs(os.path.join("reports", dataset_name), exist_ok=True)
    param_dir = os.path.join(
        "reports",
        f"params_fednova_{dataset_name}_{num_clients}clients_{skew_type}_alpha{alpha}_E{local_epochs}",
    )

    full_train, full_test, channels = load_dataset(dataset_name)

    if skew_type == "label":
        client_indices = dirichlet_noniid_partition(
            full_train, num_clients=num_clients, alpha=alpha, num_classes=10, seed=seed
        )
    elif skew_type == "quantity":
        client_indices = quantity_noniid_partition(
            full_train, num_clients=num_clients, alpha=alpha, seed=seed
        )
    else:
        raise ValueError("skew_type must be one of: 'label', 'quantity'")

    def client_fn(context: Context):
      try:
         cid = int(context.node_id)
      except (ValueError, TypeError):
         cid = 0
      cid = cid % num_clients

      train_subset = Subset(full_train, client_indices[cid])

      g = torch.Generator()
      g.manual_seed(seed + cid)

      train_loader = DataLoader(  
         train_subset,
         batch_size=64,
         shuffle=True,
         generator=g,
      )

      test_loader = DataLoader(
        full_test,
        batch_size=256,
        shuffle=False,
      )

      return CNNClient(
        train_loader,
        test_loader,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        num_channels=channels,
      ).to_client()




    dummy_model = CNN(input_channels=channels)
    param_count = sum(p.numel() for p in dummy_model.parameters())
    model_size_bytes = param_count * 4
    round_traffic_bytes = model_size_bytes * num_clients * 2

    # ✅ Send local_epochs to clients
    def fit_config(server_round: int):
        return {"local_epochs": int(local_epochs)}

    strategy = SaveFedNovaStrategy(
        save_path=param_dir,
        on_fit_config_fn=fit_config,  # ✅ IMPORTANT
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
        evaluate_fn=get_evaluate_fn(dataset_name),
        initial_parameters=ndarrays_to_parameters([val.cpu().detach().numpy() for val in dummy_model.parameters()]),
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

    rows = []
    cent_acc = [val for _, val in hist.metrics_centralized.get("accuracy", [])]
    cent_f1 = [val for _, val in hist.metrics_centralized.get("f1_score", [])]
    cent_auc = [val for _, val in hist.metrics_centralized.get("auc", [])]
    dist_min_acc = [val for _, val in hist.metrics_distributed.get("min_accuracy", [])]
    losses = [val for _, val in hist.losses_centralized]

    for i in range(len(losses)):
        round_num = i + 1
        rows.append(
            {
                "round": round_num,
                "global_loss": losses[i],
                "global_accuracy": cent_acc[i] if i < len(cent_acc) else 0.0,
                "f1_score": cent_f1[i] if i < len(cent_f1) else 0.0,
                "auc": cent_auc[i] if i < len(cent_auc) else 0.0,
                "worst_client_acc": dist_min_acc[i] if i < len(dist_min_acc) else 0.0,
                "round_traffic_bytes": round_traffic_bytes,
                "total_traffic_bytes": round_traffic_bytes * round_num,
                "total_runtime": total_runtime,
                "num_clients": num_clients,
                "dataset": dataset_name,
                "skew_type": skew_type,
                "alpha": alpha,
                "local_epochs": int(local_epochs),
                "seed": seed,  
                
            }
        )

    return rows


if __name__ == "__main__":
    run_experiment(
        dataset_name="cifar10",
        num_clients=10,
        num_rounds=20,
        seed=42,
        skew_type="label",
        alpha=0.3,
        local_epochs=5,   # ✅ change to 1, 5, 10...
    )
