import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from pathlib import Path
import numpy as np


DATASETS = ["mnist", "fashion_mnist"]

def plot_heatmap_for_dataset(dataset_name):
    print(f"--- Generating Heatmap for {dataset_name} ---")
    
    
    CSV_DIR = Path(f"reports/{dataset_name}/csv")
    FIG_DIR = Path(f"reports/{dataset_name}/figures")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

   
    files = glob(str(CSV_DIR / "results_fedavg_*_clients_*.csv"))
    
    if not files:
        print(f"No results found for {dataset_name} in {CSV_DIR}")
        return

   
    accuracy_data = {}
    max_rounds = 0

    for f in files:
        try:
            df = pd.read_csv(f)
            
         
            if "num_clients" in df.columns:
                nc = int(df["num_clients"].iloc[0])
            else:
                continue

            acc = df["global_accuracy"].tolist()
            accuracy_data[nc] = acc
            max_rounds = max(max_rounds, len(acc))
            
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not accuracy_data:
        print("No valid accuracy data found.")
        return

 
    clients = sorted(accuracy_data.keys())


    heatmap_matrix = []

    for nc in clients:
        acc = accuracy_data[nc]
   
        padded_acc = acc + [np.nan] * (max_rounds - len(acc))
        heatmap_matrix.append(padded_acc)

    heatmap_matrix = np.array(heatmap_matrix)


    plt.figure(figsize=(10, 6))
    
    sns.heatmap(
        heatmap_matrix,
        annot=True,
        fmt=".3f",
        cmap="viridis",  # 'magma' or 'inferno' are also good for accuracy
        xticklabels=[f"R{i}" for i in range(1, max_rounds + 1)],
        yticklabels=[f"{c} clients" for c in clients],
        cbar_kws={'label': 'Global Accuracy'}
    )

    plt.xlabel("Federated Rounds")
    plt.ylabel("Number of Clients")
    plt.title(f"{dataset_name.upper()}: Accuracy Heatmap")
    plt.tight_layout()


    output_path = FIG_DIR / "accuracy_heatmap_rounds_clients.png"
    plt.savefig(output_path)
    plt.close()
    
    print(f"Saved heatmap to {output_path}")

if __name__ == "__main__":
    for dataset in DATASETS:
        plot_heatmap_for_dataset(dataset)
        