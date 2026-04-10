# experiment_manager_iid_fedavg.py
from run_fedavg import run_experiment
from utils import save_csv
from datetime import datetime
import os

DATASETS = ["mnist", "fashion_mnist"]
CLIENT_SETTINGS = [2, 5, 10, 12, 15]
NUM_ROUNDS = 5
LOCAL_EPOCHS = 1          # keep explicit (IID baseline)
SEED = 42

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for dataset in DATASETS:
    print(f"\n--- Starting IID FedAvg experiments for {dataset} ---")

    output_dir = f"reports/{dataset}/csv"
    os.makedirs(output_dir, exist_ok=True)

    for nc in CLIENT_SETTINGS:
        print(f"FedAvg | {dataset} | clients={nc} | rounds={NUM_ROUNDS} | IID | local_epochs={LOCAL_EPOCHS}")

        rows = run_experiment(
            dataset_name=dataset,
            num_clients=nc,
            num_rounds=NUM_ROUNDS,
            seed=SEED,
            skew_type="iid",          # ✅ IMPORTANT: force IID
            local_epochs=LOCAL_EPOCHS
        )

        save_csv(
            rows,
            f"{output_dir}/fedavg_{dataset}_{nc}clients_iid_R{NUM_ROUNDS}_E{LOCAL_EPOCHS}_{timestamp}.csv",
        )

print("\nAll IID FedAvg experiments completed.")
