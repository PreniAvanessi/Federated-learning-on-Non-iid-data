# make_plots_epochs_comparison.py
# Compare effect of local epochs E on FedAvg/FedProx/FedNova
# Fixed: clients=10, alpha=0.3
# Vary: local_epochs = 1, 5, 10
# Skew: label and quantity (both)
#
# Outputs saved to: reports/cifar10/plots_epochs/

from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


# -----------------------------
# CONFIG
# -----------------------------
ROOT = Path("reports/cifar10/csv")
OUTDIR = Path("reports/cifar10/plots_epochs")
OUTDIR.mkdir(parents=True, exist_ok=True)

DATASET = "cifar10"
NUM_ROUNDS = 20
CLIENTS = 10
ALPHA = 0.1
EPOCHS_LIST = [1, 5, 10]
SKEW_TYPES = ["label", "quantity"]

ALGOS = {"fedavg": "FedAvg", "fedprox": "FedProx", "fednova": "FedNova"}

# Centralized baselines (adjust if you have exact values for F1/AUC)
CENTRALIZED_ACC = 0.7553
CENTRALIZED_F1  = 0.74   # approximate; update if known
CENTRALIZED_AUC = 0.98   # approximate; update if known

# Convergence target for "rounds to target"
ACC_TARGET = 0.50

# -----------------------------
# Helpers
# -----------------------------
def extract_timestamp(name: str) -> str:
    m = re.search(r"(\d{8}_\d{6})", name)
    return m.group(1) if m else ""

def choose_latest(paths):
    if not paths:
        return None
    paths = sorted(paths, key=lambda p: (extract_timestamp(p.name), p.name))
    return paths[-1]

def find_csv(algo_key: str, skew: str, E: int) -> Path | None:
    algo_key = algo_key.lower()
    e_str = str(E)

    candidates = []
    for p in ROOT.glob("*.csv"):
        n_lower = p.name.lower()

        if algo_key not in n_lower:
            continue
        if DATASET not in n_lower:
            continue
        if f"{CLIENTS}clients" not in n_lower:
            continue
        if skew not in n_lower:
            continue
        if f"alpha{ALPHA}" not in n_lower:
            continue

        # Flexible epoch matching
        epoch_vars = [f"e{e_str}", f"e={e_str}", f"epochs{e_str}", f"epoch{e_str}",
                      f"E{e_str}", f"E={e_str}", f"EPOCHS{e_str}", f"EPOCH{e_str}"]
        if not any(v in n_lower for v in epoch_vars):
            continue

        candidates.append(p)

    if not candidates:
        print(f"[NO MATCH] {algo_key.upper()} | {skew} | E={E}")
        return None

    return choose_latest(candidates)

def load_round_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["round"] <= NUM_ROUNDS].copy()
    df = df.groupby("round", as_index=False).mean(numeric_only=True).sort_values("round")
    return df

def final_row(df: pd.DataFrame) -> pd.Series:
    return df.sort_values("round").iloc[-1]

def rounds_to_target(df: pd.DataFrame, target: float) -> int | None:
    hit = df[df["global_accuracy"] >= target]
    if hit.empty:
        return None
    return int(hit.sort_values("round").iloc[0]["round"])

# -----------------------------
# Load data
# -----------------------------
data = {skew: {E: {} for E in EPOCHS_LIST} for skew in SKEW_TYPES}

for skew in SKEW_TYPES:
    for E in EPOCHS_LIST:
        for algo_key, algo_name in ALGOS.items():
            f = find_csv(algo_key, skew, E)
            if f is None:
                print(f"[MISSING] {algo_name} | {skew} | alpha={ALPHA} | clients={CLIENTS} | E={E}")
                continue
            df = load_round_df(f)
            data[skew][E][algo_name] = df
            print(f"[OK] {algo_name} | {skew} | E={E} -> {f.name}")

# -----------------------------
# Plot A: Global Accuracy vs Round (subplots by E)
# -----------------------------
for skew in SKEW_TYPES:
    fig, axes = plt.subplots(1, len(EPOCHS_LIST), figsize=(15, 4), sharey=True, sharex=True)
    if len(EPOCHS_LIST) == 1:
        axes = [axes]

    for i, (ax, E) in enumerate(zip(axes, EPOCHS_LIST)):
        for algo_name in ALGOS.values():
            if algo_name not in data[skew][E]:
                continue
            df = data[skew][E][algo_name]
            ax.plot(df["round"], df["global_accuracy"], marker="o", label=algo_name)

        # Add centralized baseline (only label once)
        ax.axhline(y=CENTRALIZED_ACC, color='black', linestyle='--', linewidth=1.5,
                   label='Centralized Baseline' if i == 0 else None)

        ax.set_title(f"E = {E}")
        ax.set_xlabel("Round")
        ax.grid(True)

    axes[0].set_ylabel("Global Accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4)
    fig.suptitle(f"{DATASET.upper()} | {skew} skew | clients={CLIENTS} | alpha={ALPHA}\n"
                 f"Global Accuracy vs Round (varying local epochs)")
    plt.tight_layout(rect=[0, 0.10, 1, 0.92])

    out = OUTDIR / f"{DATASET}_{skew}_clients{CLIENTS}_alpha{ALPHA}_epochs_accuracy_curves.png"
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print("Saved:", out)

# -----------------------------
# Plot B: Worst-client accuracy vs Round (subplots by E)
# -----------------------------
for skew in SKEW_TYPES:
    fig, axes = plt.subplots(1, len(EPOCHS_LIST), figsize=(15, 4), sharey=True, sharex=True)
    if len(EPOCHS_LIST) == 1:
        axes = [axes]

    for ax, E in zip(axes, EPOCHS_LIST):
        for algo_name in ALGOS.values():
            if algo_name not in data[skew][E]:
                continue
            df = data[skew][E][algo_name]
            ax.plot(df["round"], df["worst_client_acc"], marker="o", label=algo_name)

        ax.set_title(f"E = {E}")
        ax.set_xlabel("Round")
        ax.grid(True)

    axes[0].set_ylabel("Worst-client Accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle(f"{DATASET.upper()} | {skew} skew | clients={CLIENTS} | alpha={ALPHA}\n"
                 f"Worst-client Accuracy vs Round (varying local epochs)")
    plt.tight_layout(rect=[0, 0.10, 1, 0.92])

    out = OUTDIR / f"{DATASET}_{skew}_clients{CLIENTS}_alpha{ALPHA}_epochs_worst_client_curves.png"
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print("Saved:", out)

# -----------------------------
# Plot D: F1-Score vs Round (subplots by E)
# -----------------------------
for skew in SKEW_TYPES:
    fig, axes = plt.subplots(1, len(EPOCHS_LIST), figsize=(15, 4), sharey=True, sharex=True)
    if len(EPOCHS_LIST) == 1:
        axes = [axes]

    for i, (ax, E) in enumerate(zip(axes, EPOCHS_LIST)):
        for algo_name in ALGOS.values():
            if algo_name not in data[skew][E]:
                continue
            df = data[skew][E][algo_name]
            ax.plot(df["round"], df["f1_score"], marker="o", label=algo_name)

        # Add centralized baseline (only once)
        ax.axhline(y=CENTRALIZED_F1, color='black', linestyle='--', linewidth=1.5,
                   label='Centralized Baseline' if i == 0 else None)

        ax.set_title(f"E = {E}")
        ax.set_xlabel("Round")
        ax.grid(True)

    axes[0].set_ylabel("F1-Score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4)
    fig.suptitle(f"{DATASET.upper()} | {skew} skew | clients={CLIENTS} | alpha={ALPHA}\n"
                 f"F1-Score vs Round (varying local epochs)")
    plt.tight_layout(rect=[0, 0.10, 1, 0.92])

    out = OUTDIR / f"{DATASET}_{skew}_clients{CLIENTS}_alpha{ALPHA}_epochs_f1_curves.png"
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print("Saved F1 curves:", out)

# -----------------------------
# Plot E: AUC vs Round (subplots by E)
# -----------------------------
for skew in SKEW_TYPES:
    fig, axes = plt.subplots(1, len(EPOCHS_LIST), figsize=(15, 4), sharey=True, sharex=True)
    if len(EPOCHS_LIST) == 1:
        axes = [axes]

    for i, (ax, E) in enumerate(zip(axes, EPOCHS_LIST)):
        for algo_name in ALGOS.values():
            if algo_name not in data[skew][E]:
                continue
            df = data[skew][E][algo_name]
            ax.plot(df["round"], df["auc"], marker="o", label=algo_name)

        # Add centralized baseline (only once)
        ax.axhline(y=CENTRALIZED_AUC, color='black', linestyle='--', linewidth=1.5,
                   label='Centralized Baseline' if i == 0 else None)

        ax.set_title(f"E = {E}")
        ax.set_xlabel("Round")
        ax.grid(True)

    axes[0].set_ylabel("AUC")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4)
    fig.suptitle(f"{DATASET.upper()} | {skew} skew | clients={CLIENTS} | alpha={ALPHA}\n"
                 f"AUC vs Round (varying local epochs)")
    plt.tight_layout(rect=[0, 0.10, 1, 0.92])

    out = OUTDIR / f"{DATASET}_{skew}_clients{CLIENTS}_alpha{ALPHA}_epochs_auc_curves.png"
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print("Saved AUC curves:", out)

print("\nAll plots generated successfully!")



#increasing E on runtime impact
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# =====================
# CONFIG - UPDATED FOR ALPHA 0.3
# =====================
CSV_DIR = Path(r"C:\Users\TUF\Desktop\federated noniid project\source\reports\cifar10\csv")
OUTDIR = Path(r"C:\Users\TUF\Desktop\federated noniid project\source\reports\cifar10\plots_epochs")
OUTDIR.mkdir(parents=True, exist_ok=True)

# TARGET SETTINGS
TARGET_ALPHA = "alpha0.3"  # <--- CHANGED THIS
TARGET_CLIENTS = "10clients"

all_rows = []
csv_files = list(CSV_DIR.glob("*.csv"))

print(f"🔍 Searching for files matching {TARGET_CLIENTS} and {TARGET_ALPHA}...")

# =====================
# DATA EXTRACTION
# =====================
for f in csv_files:
    if "merged" in f.name.lower(): continue 
    try:
        name = f.name.lower()
        # Updated filter logic to catch the new alpha setting
        if TARGET_CLIENTS in name and TARGET_ALPHA in name:
            temp_df = pd.read_csv(f)
            r5 = temp_df[temp_df["round"] == 5]
            if not r5.empty:
                algo = "FedAvg" if "fedavg" in name else "FedProx" if "fedprox" in name else "FedNova"
                skew = "Label" if "label" in name else "Quantity"
                e_val = 10 if "e10" in name else 5 if "e5" in name else 1
                
                all_rows.append({
                    "skew_type": skew,
                    "local_epochs": e_val,
                    "algorithm": algo,
                    "total_runtime": r5["total_runtime"].values[0]
                })
    except Exception as e:
        print(f"⚠️ Error reading {f.name}: {e}")

# =====================
# BARPLOT GENERATION
# =====================
if not all_rows:
    print(f"❌ No matching data found for {TARGET_ALPHA}. Check your CSV filenames!")
else:
    df = pd.DataFrame(all_rows).groupby(["skew_type", "local_epochs", "algorithm"], as_index=False).mean()

    for skew in df["skew_type"].unique():
        sub = df[df["skew_type"] == skew]
        epochs = sorted(sub["local_epochs"].unique())
        algos = ["FedAvg", "FedProx", "FedNova"]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(epochs))
        width = 0.25
        colors = {"FedAvg": "#1f77b4", "FedProx": "#2ca02c", "FedNova": "#ff7f0e"}

        for i, algo in enumerate(algos):
            algo_sub = sub[sub["algorithm"] == algo].sort_values("local_epochs")
            if not algo_sub.empty:
                ax.bar(x + i*width, algo_sub["total_runtime"], width, 
                        label=algo, color=colors[algo], edgecolor='black', alpha=0.8)

        # UPDATED LABELS FOR ALPHA 0.3
        ax.set_xlabel("Local Epochs (E)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Total Runtime (Seconds)", fontsize=12, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels([f"E={e}" for e in epochs])
        ax.set_title(f"Runtime Comparison: {skew} Skew\n(K=10, Alpha=0.3, 5 Rounds)", fontsize=14)
        
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend(title="Algorithm", loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        # Update filename to prevent overwriting your alpha 0.1 plots
        save_path = OUTDIR / f"runtime_bar_{skew.lower()}_alpha0.3.png"
        plt.savefig(save_path, dpi=300)
        print(f"✅ Saved alpha=0.3 barplot to: {save_path}")

    plt.show()