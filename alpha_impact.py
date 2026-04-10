import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# =====================
# SETTINGS
# =====================
ROOT = Path("reports/cifar10/csv/5 rounds")
DATASET = "cifar10"
SKEW_TYPE = "label"      # Change to "label" or "quantity"
CLIENTS = 10
LOCAL_EPOCHS = 5
ALPHAS = [0.1, 0.3, 0.5]
NUM_ROUNDS = 5 
ACC_TARGET = 0.50

OUTDIR = Path(f"reports/cifar10/plots_{SKEW_TYPE}_impact")
OUTDIR.mkdir(parents=True, exist_ok=True)

ALGORITHMS = {"fedavg": "FedAvg", "fedprox": "FedProx", "fednova": "FedNova"}

# =====================
# LOAD DATA
# =====================
records = []
for csv_file in ROOT.glob("*.csv"):
    name = csv_file.name.lower()
    if DATASET not in name or f"{CLIENTS}clients" not in name or SKEW_TYPE not in name:
        continue
    if f"e{LOCAL_EPOCHS}" not in name:
        continue

    algo = next((ALGORITHMS[k] for k in ALGORITHMS if name.startswith(k)), None)
    alpha = next((a for a in ALPHAS if f"alpha{a}" in name), None)
    
    if algo is None or alpha is None:
        continue

    try:
        temp_df = pd.read_csv(csv_file)
        # Convergence calculation
        hit = temp_df[temp_df["global_accuracy"] >= ACC_TARGET]
        speed = int(hit["round"].min()) if not hit.empty else NUM_ROUNDS + 1
        
        df_final = temp_df[temp_df["round"] == NUM_ROUNDS]
        if not df_final.empty:
            row = df_final.iloc[0]
            records.append({
                "Algorithm": algo,
                "Alpha": float(alpha),
                "Global Acc": round(float(row.get("global_accuracy", 0.0)), 4),
                "Worst Acc": round(float(row.get("worst_client_acc", 0.0)), 4),
                "F1-Score": round(float(row.get("f1_score", 0.0)), 4),
                "Rounds to Target": speed
            })
    except Exception as e:
        print(f"Error: {e}")

if not records:
    print(f"❌ No records found.")
    sys.exit()

df = pd.DataFrame(records).sort_values(["Alpha", "Algorithm"])

baseline_df = pd.read_csv(r"reports\baseline\cifar10\baseline_cifar10_log.csv")


# Global Plot Styling
plt.rcParams.update({'font.size': 14})




# =====================
# FIGURE 1: ACCURACY METRICS (Line Plots)
# =====================
# =====================
# FIGURE 1: ACCURACY METRICS (Merged with Baseline)
# =====================
fig1, axes1 = plt.subplots(1, 3, figsize=(24, 8)) 
metrics = [("Global Acc", "Global Accuracy"), 
           ("Worst Acc", "Worst-client Accuracy"), 
           ("F1-Score", "F1-Score")]

# These values come from your baseline log
BASELINE_ACC = 0.7553  
BASELINE_F1  = 0.7553

for i, (col, title) in enumerate(metrics):
    # 1. Plot Federated Learning Algorithms
    for algo in df["Algorithm"].unique():
        sub = df[df["Algorithm"] == algo].sort_values("Alpha")
        axes1[i].plot(sub["Alpha"], sub[col], marker="o", linewidth=4, markersize=12, label=algo)
    
    # 2. Add Centralized Reference Lines (The "Ceiling")
    if "Accuracy" in title:
        axes1[i].axhline(y=BASELINE_ACC, color='black', linestyle='--', linewidth=2, label='Centralized Baseline')
    elif "F1" in title:
        axes1[i].axhline(y=BASELINE_F1, color='black', linestyle='--', linewidth=2, label='Centralized Baseline')

    axes1[i].set_title(title, fontweight='bold', fontsize=22, pad=20)
    axes1[i].set_xlabel("Dirichlet α", fontsize=18)
    axes1[i].set_xticks(ALPHAS)
    axes1[i].grid(True, linestyle='--', alpha=0.6)

axes1[0].set_ylabel("Metric Value", fontsize=18)
axes1[-1].legend(title="Algorithm", fontsize=14, loc='lower right')
fig1.suptitle(f"CIFAR-10 FL vs Centralized Baseline | {SKEW_TYPE.upper()} Skew", fontsize=28, fontweight='bold', y=1.02)

# 3. SAVE THE FIGURE (Must be done BEFORE plt.show())
path1 = OUTDIR / f"{DATASET}_{SKEW_TYPE}_accuracy_with_baseline.png"
fig1.savefig(path1, dpi=300, bbox_inches='tight')
print(f"✅ Final Accuracy plot with Baseline saved to: {path1}")

plt.show()
# =====================
# FIGURE 2: CONVERGENCE SPEED (Bar Plot)
# =====================
fig2, ax2 = plt.subplots(figsize=(12, 8))
x_indices = np.arange(len(ALPHAS))
width = 0.25
algos = df["Algorithm"].unique()

for j, algo in enumerate(algos):
    sub = df[df["Algorithm"] == algo].sort_values("Alpha")
    heights = sub["Rounds to Target"].clip(upper=NUM_ROUNDS)
    rects = ax2.bar(x_indices + (j * width) - width, heights, width, label=algo)
    
    # Label DNF for failed runs
    for idx, rect in enumerate(rects):
        if sub.iloc[idx]["Rounds to Target"] > NUM_ROUNDS:
            ax2.annotate('DNF', xy=(rect.get_x() + rect.get_width() / 2, NUM_ROUNDS),
                            xytext=(0, 5), textcoords="offset points",
                            ha='center', va='bottom', color='red', fontsize=14, fontweight='bold')

ax2.set_title(f"Convergence Speed: Rounds to reach {int(ACC_TARGET*100)}% Acc", fontsize=22, fontweight='bold', pad=20)
ax2.set_xticks(x_indices)
ax2.set_xticklabels(ALPHAS)
ax2.set_xlabel("Dirichlet α", fontsize=18)
ax2.set_ylabel("Communication Rounds", fontsize=18)
ax2.set_ylim(0, NUM_ROUNDS)
ax2.set_yticks(range(0, NUM_ROUNDS + 1))
ax2.grid(axis='y', linestyle='--', alpha=0.6)
ax2.legend(title="Algorithm", loc='upper right')

# Save Figure 2
path2 = OUTDIR / f"{DATASET}_{SKEW_TYPE}_convergence_speed.png"
fig2.savefig(path2, dpi=300, bbox_inches='tight')
print(f"✅ Convergence plot saved to: {path2}")

plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# =====================
# 1. DEFINE BASELINE FROM YOUR LOG
# =====================
# Based on your provided CSV data:
BASELINE_ACC = 0.7553  # Max test_accuracy at epoch 17
BASELINE_F1  = 0.7553  # Max test_f1 at epoch 17

# =====================
# 2. GENERATE FINAL PLOTS
# =====================
fig, axes = plt.subplots(1, 3, figsize=(24, 8))
metrics = [("Global Acc", "Global Accuracy"), 
           ("Worst Acc", "Worst-client Accuracy"), 
           ("F1-Score", "F1-Score")]

for i, (col, title) in enumerate(metrics):
    # Plot Federated Learning Algorithms (FedAvg, FedProx, FedNova)
    for algo in df["Algorithm"].unique():
        sub = df[df["Algorithm"] == algo].sort_values("Alpha")
        axes[i].plot(sub["Alpha"], sub[col], marker="o", linewidth=4, markersize=12, label=algo)
    
    # Add Centralized Reference Lines
    if "Accuracy" in title:
        # Drawing the line across all accuracy plots to show the "ceiling"
        axes[i].axhline(y=BASELINE_ACC, color='black', linestyle='--', linewidth=2, label='Centralized Baseline')
    elif "F1" in title:
        axes[i].axhline(y=BASELINE_F1, color='black', linestyle='--', linewidth=2, label='Centralized Baseline')

    axes[i].set_title(title, fontweight='bold', fontsize=22, pad=20)
    axes[i].set_xlabel("Dirichlet α", fontsize=18)
    axes[i].set_xticks([0.1, 0.3, 0.5])
    axes[i].grid(True, linestyle='--', alpha=0.6)

axes[0].set_ylabel("Metric Value", fontsize=18)
axes[-1].legend(title="Algorithm", fontsize=14, loc='lower right')

plt.tight_layout()
plt.show()


#NEW: PLOT ACCURACY CURVES OVER ROUNDS (per α)
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# =====================
# SETTINGS
# =====================
ROOT = Path("reports/cifar10/csv/5 rounds")  # ← change if your folder is different
DATASET = "cifar10"
SKEW_TYPE = "quantity"           # Change to "quantity" when needed
CLIENTS = 10
LOCAL_EPOCHS = 5
ALPHAS = [0.1, 0.3, 0.5]
NUM_ROUNDS = 5

OUTDIR = Path(f"reports/cifar10/plots_{SKEW_TYPE}_impact")
OUTDIR.mkdir(parents=True, exist_ok=True)

ALGORITHMS = {
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "fednova": "FedNova"
}

# Baseline values (adjust if needed)
BASELINE_ACC = 0.7553   # global accuracy from your centralized run

# Plot styling
COLORS = {"FedAvg": "tab:blue", "FedProx": "tab:green", "FedNova": "tab:orange"}
MARKERS = {"FedAvg": "o", "FedProx": "s", "FedNova": "^"}
LINEWIDTH = 2.8
MARKERSIZE = 9

# =====================
# LOAD FULL PER-ROUND DATA
# =====================
full_data = {}  # algo → alpha → df_rounds

print("Searching for CSV files in:", ROOT)

for csv_file in ROOT.glob("*.csv"):
    name = csv_file.name.lower()
    print(f"Checking file: {csv_file.name}")

    if not all(x in name for x in [DATASET.lower(), f"{CLIENTS}clients", SKEW_TYPE]):
        print("  → skipped (doesn't match dataset/clients/skew)")
        continue
    if f"e{LOCAL_EPOCHS}" not in name:
        print("  → skipped (wrong local epochs)")
        continue

    algo_key = next((k for k in ALGORITHMS if name.startswith(k)), None)
    if algo_key is None:
        print("  → skipped (unknown algorithm)")
        continue
    algo = ALGORITHMS[algo_key]

    alpha_str = next((str(a) for a in ALPHAS if f"alpha{a}" in name), None)
    if alpha_str is None:
        print("  → skipped (unknown alpha)")
        continue
    alpha = float(alpha_str)

    try:
        df_rounds = pd.read_csv(csv_file)
        df_rounds["round"] = df_rounds["round"].astype(int)
        full_data.setdefault(algo, {})[alpha] = df_rounds
        print(f"  → SUCCESS: loaded {algo} α={alpha} ({len(df_rounds)} rows)")
    except Exception as e:
        print(f"  → ERROR loading {csv_file}: {e}")

if not full_data:
    print("\n❌ No matching CSV files found. Check:")
    print("   - Folder path:", ROOT)
    print("   - Filenames contain:", DATASET, f"{CLIENTS}clients", SKEW_TYPE, f"e{LOCAL_EPOCHS}", "alpha0.1/0.3/0.5")
    print("   - Algorithms start with fedavg/fedprox/fednova")
    sys.exit()

print("\nLoaded data summary:")
for algo, alphas_dict in full_data.items():
    print(f"  {algo}: alphas = {list(alphas_dict.keys())}")

# =====================
# PLOT FUNCTION
# =====================
def plot_metric_over_rounds(metric_key, metric_label, baseline_val=None):
    fig, axes = plt.subplots(1, len(ALPHAS), figsize=(18, 6), sharey=True)
    fig.suptitle(f"{metric_label} over Communication Rounds\n{SKEW_TYPE.capitalize()} Skew, K={CLIENTS}, E={LOCAL_EPOCHS}",
                 fontsize=20, fontweight='bold', y=1.03)

    for idx, alpha in enumerate(sorted(ALPHAS)):
        ax = axes[idx]

        for algo in ["FedAvg", "FedProx", "FedNova"]:
            if alpha in full_data.get(algo, {}):
                df = full_data[algo][alpha]
                df = df[df["round"] <= NUM_ROUNDS].sort_values("round")
                ax.plot(df["round"], df[metric_key],
                        color=COLORS.get(algo),
                        marker=MARKERS.get(algo),
                        linewidth=LINEWIDTH,
                        markersize=MARKERSIZE,
                        label=algo)

        if baseline_val is not None:
            ax.axhline(y=baseline_val, color="black", linestyle="--", linewidth=2.5,
                       label="Centralized Baseline")
            print(f"Baseline added at y={baseline_val} for {metric_label}, α={alpha}")

        ax.set_title(f"α = {alpha}", fontsize=16, pad=12)
        ax.set_xlabel("Communication Round", fontsize=14)
        ax.set_ylabel(metric_label, fontsize=14)
        ax.set_xticks(range(1, NUM_ROUNDS+1))
        ax.set_xlim(0.8, NUM_ROUNDS + 0.2)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=11, loc="lower right")

    # Consistent y-limits + force include baseline if present
    all_values = []
    for ax in axes:
        for line in ax.get_lines():
            if line.get_label() != "Centralized Baseline":
                all_values.extend(line.get_ydata())
    if all_values:
        y_min = max(0, min(all_values) - 0.03)
        y_max = max(all_values) + 0.03
        if baseline_val is not None:
            y_max = max(y_max, baseline_val + 0.03)  # ensure baseline visible
        for ax in axes:
            ax.set_ylim(y_min, y_max)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = OUTDIR / f"{DATASET}_{SKEW_TYPE}_{metric_key}_vs_rounds_{NUM_ROUNDS}rounds.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved: {save_path}")
    plt.show()
    plt.close(fig)

# =====================
# GENERATE PLOTS
# =====================
print("\nGenerating plots...")

# 1. Global Accuracy (with baseline)
plot_metric_over_rounds("global_accuracy", "Global Accuracy", baseline_val=BASELINE_ACC)

# 2. Worst-Client Accuracy (no baseline)
plot_metric_over_rounds("worst_client_acc", "Worst-Client Accuracy", baseline_val=None)

print("Done!")