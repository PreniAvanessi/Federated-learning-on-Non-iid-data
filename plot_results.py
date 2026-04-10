import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re
from pathlib import Path


DATASET = "fashion_mnist"  
CSV_DIR = Path(f"reports/{DATASET}/csv")
FIG_DIR = Path(f"reports/{DATASET}/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


TARGETS = { "mnist": 0.95, "fashion_mnist": 0.85 }
TARGET_ACC = TARGETS.get(DATASET, 0.80)
MAX_ROUNDS = 5  # <--- Strict Limit for the experiment

def get_latest_csvs(csv_dir):
    """Finds the latest results file for each client count."""
    files = glob.glob(str(csv_dir / "results_fedavg_*_clients_*.csv"))
    latest_files = {}
    for f in files:
       
        match = re.search(r"results_fedavg_(\d+)_clients_(\d+_\d+).csv", f)
        if match:
            n_clients = int(match.group(1))
            timestamp = match.group(2)
        
            if n_clients not in latest_files or timestamp > latest_files[n_clients][0]:
                latest_files[n_clients] = (timestamp, f)
    return latest_files

latest_data = get_latest_csvs(CSV_DIR)

if not latest_data:
    print(f" No CSV files found in {CSV_DIR}")
    print(f"   Please check if experiment_manager.py ran successfully.")
else:
    print(f"--- Generating All Plots (Cleaning data > 5 rounds) for {DATASET} ---")
    
    metrics_to_plot = {
        "Accuracy": "global_accuracy",
        "Loss": "global_loss",
        "F1-Score": "f1_score",
        "AUC": "auc",
        "Worst Client Accuracy": "worst_client_accuracy",  # Fixed column name if needed
        "Total Communication (MB)": "total_traffic_bytes"
    }

    
    figs = {}
    for title in metrics_to_plot.keys():
        fig = plt.figure(figsize=(10, 6))
        figs[title] = fig
        plt.title(f"{DATASET}: {title}")
        plt.xlabel("Round")
        plt.ylabel(title)
        plt.grid(True)
        # Add target line for Accuracy plot
        if title == "Accuracy":
             plt.axhline(y=TARGET_ACC, color='k', linestyle='--', alpha=0.5, label=f"Target ({TARGET_ACC})")

    convergence_results = []
    runtime_results = [] 
    markers = ['o', 's', '^', 'D', 'x', '*'] 

  
    for i, n_clients in enumerate(sorted(latest_data.keys())):
        timestamp, filepath = latest_data[n_clients]
        marker = markers[i % len(markers)]
        
        try:
            df = pd.read_csv(filepath)
            
           
            if "round" in df.columns:
                
                df = df[df["round"] <= MAX_ROUNDS]
            
            if "global_accuracy" in df.columns:
               
                df = df[df["global_accuracy"] > 0.0]

            if df.empty:
                print(f"    Data for {n_clients} clients became empty after filtering!")
                continue

           
            for title, col_name in metrics_to_plot.items():
                
                if col_name not in df.columns:
                    if col_name == "worst_client_accuracy" and "worst_client_acc" in df.columns:
                        col_name = "worst_client_acc"
                
                if col_name in df.columns:
                    plt.figure(figs[title].number) # Activate the correct figure
                    
                    y_values = df[col_name]
                   
                    if col_name == "total_traffic_bytes":
                        y_values = y_values / (1024 * 1024)
                    
                    plt.plot(df["round"], y_values, marker=marker, label=f"{n_clients} Clients")

       
            if "total_runtime" in df.columns:
                
                runtime = df["total_runtime"].max()
                runtime_results.append({"Clients": n_clients, "Runtime": runtime})

       
            if "global_accuracy" in df.columns:
                converged_rows = df[df["global_accuracy"] >= TARGET_ACC]
                if not converged_rows.empty:
                    convergence_results.append({"Clients": n_clients, "Rounds": converged_rows.iloc[0]["round"]})
                else:
                   
                    convergence_results.append({"Clients": n_clients, "Rounds": MAX_ROUNDS + 1})

        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    for title, fig in figs.items():
        plt.figure(fig.number)
        plt.legend()
        plt.tight_layout()
        safe_name = title.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        save_path = FIG_DIR / f"combined_{safe_name}.png"
        plt.savefig(save_path)
        plt.close(fig)
        print(f"   ✅ Saved {title} plot to {save_path.name}")

    
    if convergence_results:
        df_conv = pd.DataFrame(convergence_results)
        plt.figure(figsize=(8, 6))
        bars = plt.bar(df_conv["Clients"].astype(str), df_conv["Rounds"], color="orange")
        plt.xlabel("Number of Clients")
        plt.ylabel(f"Rounds to Reach {TARGET_ACC*100}%")
        plt.title(f"Convergence Speed: {DATASET}")
        plt.grid(axis="y")
        
     
        for bar in bars:
            height = bar.get_height()
            label = f"{int(height)}" if height <= MAX_ROUNDS else "Not Reached"
            plt.text(bar.get_x() + bar.get_width()/2., height, label, ha='center', va='bottom')
            
        plt.savefig(FIG_DIR / "convergence_comparison.png")
        plt.close()
        print(f"   ✅ Saved Convergence plot")

    
    if runtime_results:
        df_run = pd.DataFrame(runtime_results)
        plt.figure(figsize=(8, 6))
        plt.plot(df_run["Clients"].astype(str), df_run["Runtime"], marker='o', linestyle='-', color='purple', linewidth=2)
        plt.xlabel("Number of Clients")
        plt.ylabel("Total Runtime (seconds)")
        plt.title(f"Scalability: Runtime vs Clients ({DATASET})")
        plt.grid(True)
        plt.savefig(FIG_DIR / "scalability_runtime.png")
        plt.close()
        print(f"   ✅ Saved Runtime plot")

    print(f"Done! All plots saved to {FIG_DIR}")