import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Settings
# =========================
DATASET = "cifar10"
CSV_DIR = f"reports/{DATASET}/csv"
OUT_DIR = f"reports/{DATASET}/plots_mu_impact"
os.makedirs(OUT_DIR, exist_ok=True)

MU_VALUES = [0.001, 0.01, 0.1]
SEEDS = [43, 44, 45, 46]

K_FIXED = 35
ALPHA_FIXED = 0.1
LOCAL_EPOCHS = 5
SKEW_TYPES = ["label", "quantity"]
NUM_ROUNDS = 30
CENTRALIZED_ACC = 0.7553

# =========================
# Load CSVs
# =========================
all_files = glob.glob(os.path.join(CSV_DIR, "fedprox_*.csv"))
if not all_files:
    raise FileNotFoundError(f"No FedProx CSV files found in: {CSV_DIR}")

dfs = []
required_cols = {
    "round",
    "global_accuracy",
    "num_clients",
    "skew_type",
    "alpha",
    "local_epochs",
    "seed",
    "proximal_mu",
}

for f in all_files:
    try:
        df = pd.read_csv(f)
    except Exception as e:
        print(f"Skipping unreadable file {f}: {e}")
        continue

    if not required_cols.issubset(df.columns):
        print(f"Skipping {f}: missing required columns")
        continue

    dfs.append(df)

if not dfs:
    raise ValueError("No usable FedProx CSV files found.")

data = pd.concat(dfs, ignore_index=True)

# =========================
# Clean types
# =========================
data["round"] = data["round"].astype(int)
data["num_clients"] = data["num_clients"].astype(int)
data["seed"] = data["seed"].astype(int)
data["local_epochs"] = data["local_epochs"].astype(int)
data["alpha"] = data["alpha"].astype(float)
data["proximal_mu"] = data["proximal_mu"].astype(float)

# =========================
# Helpers
# =========================
def pm(mean, std, digits=4):
    if pd.isna(mean):
        return ""
    if pd.isna(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"

def last_nonzero_worst(df_run: pd.DataFrame) -> float:
    if "worst_client_acc" not in df_run.columns:
        return np.nan

    s = df_run.sort_values("round")["worst_client_acc"].astype(float).values
    if len(s) == 0:
        return np.nan

    idx = len(s) - 1
    while idx >= 0 and np.isclose(s[idx], 0.0):
        idx -= 1

    if idx < 0:
        return float(s[-1])
    return float(s[idx])

def run_summaries(df_run: pd.DataFrame) -> pd.Series:
    df_run = df_run[df_run["round"] <= NUM_ROUNDS].sort_values("round")

    return pd.Series({
        "final_accuracy": float(df_run["global_accuracy"].iloc[-1]),
        "best_accuracy": float(df_run["global_accuracy"].max()),
        "final_f1": float(df_run["f1_score"].iloc[-1]) if "f1_score" in df_run.columns else np.nan,
        "final_auc": float(df_run["auc"].iloc[-1]) if "auc" in df_run.columns else np.nan,
        "final_worst_client_acc": last_nonzero_worst(df_run),
    })

# =========================
# Run once for each skew type
# =========================
for SKEW_TYPE in SKEW_TYPES:
    subset = data[
        (data["num_clients"] == K_FIXED) &
        (data["alpha"] == ALPHA_FIXED) &
        (data["local_epochs"] == LOCAL_EPOCHS) &
        (data["skew_type"] == SKEW_TYPE) &
        (data["seed"].isin(SEEDS)) &
        (data["proximal_mu"].isin(MU_VALUES))
    ].copy()

    if subset.empty:
        print(f"No rows found for skew_type={SKEW_TYPE}")
        continue

    print(f"\nProcessing {SKEW_TYPE} skew...")
    print("Shape:", subset.shape)
    print("Mu values found:", sorted(subset["proximal_mu"].unique().tolist()))

    # Per-run summary
    per_run = (
        subset.groupby(["proximal_mu", "seed"], group_keys=False)
        .apply(run_summaries)
        .reset_index()
    )

    # Aggregate across seeds
    agg = per_run.groupby("proximal_mu").agg(
        final_acc_mean=("final_accuracy", "mean"),
        final_acc_std=("final_accuracy", "std"),
        best_acc_mean=("best_accuracy", "mean"),
        best_acc_std=("best_accuracy", "std"),
        final_f1_mean=("final_f1", "mean"),
        final_f1_std=("final_f1", "std"),
        final_auc_mean=("final_auc", "mean"),
        final_auc_std=("final_auc", "std"),
        final_worst_mean=("final_worst_client_acc", "mean"),
        final_worst_std=("final_worst_client_acc", "std"),
    ).reset_index()

    agg["Final Acc"] = agg.apply(lambda r: pm(r["final_acc_mean"], r["final_acc_std"]), axis=1)
    agg["Best Acc"] = agg.apply(lambda r: pm(r["best_acc_mean"], r["best_acc_std"]), axis=1)
    agg["F1 Score"] = agg.apply(lambda r: pm(r["final_f1_mean"], r["final_f1_std"]), axis=1)
    agg["AUC"] = agg.apply(lambda r: pm(r["final_auc_mean"], r["final_auc_std"]), axis=1)
    agg["Worst-client Acc"] = agg.apply(lambda r: pm(r["final_worst_mean"], r["final_worst_std"]), axis=1)

    summary = agg[
        ["proximal_mu", "Final Acc", "Best Acc", "F1 Score", "AUC", "Worst-client Acc"]
    ].sort_values("proximal_mu")

    # Save formatted table
    summary_path = os.path.join(OUT_DIR, f"fedprox_mu_summary_{SKEW_TYPE}.csv")
    summary.to_csv(summary_path, index=False)

    # Save numeric table
    numeric_path = os.path.join(OUT_DIR, f"fedprox_mu_numeric_summary_{SKEW_TYPE}.csv")
    agg.to_csv(numeric_path, index=False)

    print("Saved:", summary_path)
    print("Saved:", numeric_path)
    print(summary.to_string(index=False))

    # Plot
    plt.figure(figsize=(8, 5))

    for mu in sorted(subset["proximal_mu"].unique()):
        sub_mu = subset[subset["proximal_mu"] == mu].copy()

        pivot = sub_mu.pivot_table(
            index="round",
            columns="seed",
            values="global_accuracy",
            aggfunc="mean"
        ).sort_index()

        if pivot.empty:
            continue

        mean = pivot.mean(axis=1)
        std = pivot.std(axis=1).fillna(0)

        plt.plot(mean.index, mean.values, linewidth=2, label=f"μ={mu}")
        plt.fill_between(
            mean.index,
            (mean - std).values,
            (mean + std).values,
            alpha=0.2
        )

    plt.axhline(
        y=CENTRALIZED_ACC,
        color="black",
        linestyle="--",
        linewidth=2,
        label="Centralized CNN"
    )

    plt.xlabel("Communication Round")
    plt.ylabel("Global Accuracy")
    plt.title(f"FedProx Sensitivity to μ ({SKEW_TYPE} skew, K={K_FIXED}, α={ALPHA_FIXED}, E={LOCAL_EPOCHS})")
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(
        OUT_DIR,
        f"fedprox_mu_acc_vs_round_{SKEW_TYPE}_K{K_FIXED}_alpha{ALPHA_FIXED}_E{LOCAL_EPOCHS}.png"
    )
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved:", plot_path)

print("\nDone.")