import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from pathlib import Path
import os

# Define datasets to process
DATASETS = ["mnist", "fashion_mnist"]

def plot_comm_for_dataset(dataset_name):
    print(f"--- Plotting Stats for {dataset_name} ---")
  
    CSV_DIR = Path(f"reports/{dataset_name}/csv")
    FIG_DIR = Path(f"reports/{dataset_name}/figures")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

 
    files = glob(str(CSV_DIR / "*.csv"))
    
    if not files:
        print(f"No CSV files found in {CSV_DIR}")
        return

    all_data = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if "dataset" in df.columns:
                df = df[df["dataset"] == dataset_name]
            
            if not df.empty:
                all_data.append(df)
        except Exception as e:
            print(f"Skipping file {f}: {e}")

    if not all_data:
        print(f"No matching data found for {dataset_name}")
        return

    full_df = pd.concat(all_data, ignore_index=True)

 
    comm_col = "bytes_transmitted_per_round"
    if comm_col not in full_df.columns:
        print(f"Error: Column '{comm_col}' not found. Available columns: {list(full_df.columns)}")
        return

   
    client_groups = full_df.groupby("num_clients")
    
    cumulative_data_map = {}
    total_comm_data = {}  #

    for num_clients, group in client_groups:
        # Sort by round
        group = group.sort_values("round")
        
        # Get Y-axis data (Bytes -> MB)
        mb_data = group[comm_col] / (1024 * 1024)
        
        # Calculate Cumulative Sum for the Line Plot
        cumsum_data = np.cumsum(mb_data).tolist()
        cumulative_data_map[num_clients] = cumsum_data
        
       
        total_comm_data[num_clients] = cumsum_data[-1]




   
    if cumulative_data_map:
        plt.figure(figsize=(10, 6))
        
        for nc in sorted(cumulative_data_map.keys()):
            data = cumulative_data_map[nc]
            rounds = range(1, len(data) + 1)
            plt.plot(rounds, data, marker="o", label=f"{nc} clients")

        plt.xlabel("Federated Round")
        plt.ylabel("Cumulative Data Transmitted (MB)")
        plt.title(f"{dataset_name.upper()}: Cumulative Communication Cost")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig(FIG_DIR / "cumulative_communication.png")
        plt.close()
        print(f"Saved cumulative_communication.png to {FIG_DIR}")






    if total_comm_data:
        plt.figure(figsize=(8, 6))
        clients = sorted(total_comm_data.keys())
        totals = [total_comm_data[c] for c in clients]

    
        plt.bar(clients, totals, color='skyblue', edgecolor='black', width=1.5) 
        
        plt.xlabel("Number of Clients")
        plt.ylabel("Total Data Transmitted (MB)")
        plt.title(f"{dataset_name.upper()}: Total Communication Traffic")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
  
        plt.xticks(clients)
        
        plt.tight_layout()
        plt.savefig(FIG_DIR / "total_communication.png")
        plt.close()
        print(f"Saved total_communication.png to {FIG_DIR}")

if __name__ == "__main__":
    for dataset in DATASETS:
        plot_comm_for_dataset(dataset)