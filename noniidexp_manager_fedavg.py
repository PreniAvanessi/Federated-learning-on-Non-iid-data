# experiment_manager_cifar10_noniid.py
from datetime import datetime
import os

from utils import save_csv
from run_fedavg import run_experiment as run_fedavg

DATASET = "cifar10"
CLIENT_SETTINGS = [35]
SKEW_TYPES = ["label", "quantity"]
ALPHAS = [0.1]
EPOCHS = [5]
NUM_ROUNDS = 30
SEEDS = [43,44,45,46]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"reports/{DATASET}/csv"
os.makedirs(output_dir, exist_ok=True)

for seed in SEEDS:
    for skew_type in SKEW_TYPES:
        for alpha in ALPHAS:
            for nc in CLIENT_SETTINGS:
                for E in EPOCHS:
                    print(
                        f"Running FedAvg | seed={seed} | {DATASET} | clients={nc} | rounds={NUM_ROUNDS} "
                        f"| skew={skew_type} | alpha={alpha} | local_epochs={E}"
                    )

                    rows = run_fedavg(
                        dataset_name=DATASET,
                        num_clients=nc,
                        num_rounds=NUM_ROUNDS,
                        seed=seed,  # ✅ single int seed
                        skew_type=skew_type,
                        alpha=alpha,
                        local_epochs=E,
                    )

                    save_csv(
                        rows,
                        f"{output_dir}/fedavg_{DATASET}_{nc}clients_{skew_type}_alpha{alpha}_E{E}_seed{seed}_{timestamp}.csv",
                    )

print("Done.")
