from datetime import datetime
import os

from utils import save_csv
from run_fednova import run_experiment as run_fednova

DATASET = "cifar10"
CLIENT_SETTINGS = [35]
EPOCHS = [1,10]
NUM_ROUNDS = 30
SEEDS = [43,44,45,46]

SKEW_TYPES = [ "quantity","label"]
ALPHAS = [0.1]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"reports/{DATASET}/csv"
os.makedirs(output_dir, exist_ok=True)

for seed in SEEDS:
    for skew_type in SKEW_TYPES:
        for alpha in ALPHAS:
            for nc in CLIENT_SETTINGS:
                for E in EPOCHS:

                    print(
                        f"FedNova | seed={seed} | {DATASET} | clients={nc} | rounds={NUM_ROUNDS} "
                        f"| skew={skew_type} | alpha={alpha} | local_epochs={E}"
                    )

                    rows = run_fednova(
                        dataset_name=DATASET,
                        num_clients=nc,
                        num_rounds=NUM_ROUNDS,
                        seed=seed,   # ✅ correct
                        skew_type=skew_type,
                        alpha=alpha,
                        local_epochs=E,
                    )

                    save_csv(
                        rows,
                        f"{output_dir}/fednova_{DATASET}_{nc}clients_{skew_type}_alpha{alpha}_E{E}_seed{seed}_{timestamp}.csv",
                    )

print("All FedNova experiments completed.")