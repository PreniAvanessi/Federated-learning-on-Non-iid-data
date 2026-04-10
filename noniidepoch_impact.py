

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# Settings (edit here)
# -----------------------
DATASET = "cifar10"
CSV_DIR = f"reports/{DATASET}/csv"   # where raw per-run CSVs are
OUT_DIR = f"reports/{DATASET}/plots_epochs_impact"
os.makedirs(OUT_DIR, exist_ok=True)

ALPHA = 0.1
K_FIXED = 35
EPOCHS = [1, 5, 10]
SEEDS = [43, 44, 45, 46]
SKEW_TYPES = None            
ALGORITHMS = None          


IGNORE_TRAILING_WORST_ZERO = True


CENTRALIZED_ACC = 0.755


all_files = glob.glob(os.path.join(CSV_DIR, "*.csv"))
if not all_files:
    raise FileNotFoundError(f"No CSV files found in: {CSV_DIR}")

dfs = []
required_cols = {
    "round",
    "global_accuracy",
    "num_clients",
    "skew_type",
    "alpha",
    "local_epochs",
    "seed",
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

    base = os.path.basename(f)
    algo = base.split("_", 1)[0].lower()
    df["algorithm"] = algo
    df["source_file"] = base
    dfs.append(df)

if not dfs:
    raise ValueError(
        f"No usable CSVs found in {CSV_DIR}.\n"
        "Either files are missing or required columns are not present."
    )

data = pd.concat(dfs, ignore_index=True)

# -----------------------
# Filter for this epoch-impact slice
# -----------------------
data["round"] = data["round"].astype(int)
data["num_clients"] = data["num_clients"].astype(int)
data["seed"] = data["seed"].astype(int)
data["local_epochs"] = data["local_epochs"].astype(int)
data["alpha"] = data["alpha"].astype(float)

data = data[
    (data["alpha"] == float(ALPHA)) &
    (data["num_clients"] == int(K_FIXED)) &
    (data["local_epochs"].isin(EPOCHS)) &
    (data["seed"].isin(SEEDS))
].copy()

if SKEW_TYPES is not None:
    data = data[data["skew_type"].isin(SKEW_TYPES)].copy()

if ALGORITHMS is not None:
    data = data[data["algorithm"].isin([a.lower() for a in ALGORITHMS])].copy()

if data.empty:
    raise ValueError(
        "After filtering, no rows remain.\n"
        f"ALPHA={ALPHA}, K_FIXED={K_FIXED}, EPOCHS={EPOCHS}, SEEDS={SEEDS}\n"
        f"CSV_DIR={CSV_DIR}\n"
        "Check that your CSVs contain these exact values."
    )

print("Filtered data shape:", data.shape)
print("Algorithms found:", sorted(data["algorithm"].dropna().unique().tolist()))
print("Skew types found:", sorted(data["skew_type"].dropna().unique().tolist()))

# -----------------------
# Helpers
# -----------------------
def pm(mean, std, digits=4):
    if pd.isna(mean):
        return ""
    if pd.isna(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"

def last_nonzero_worst(df_run: pd.DataFrame) -> float:
    """Return last worst_client_acc, optionally ignoring trailing zeros."""
    if "worst_client_acc" not in df_run.columns:
        return np.nan

    s = df_run.sort_values("round")["worst_client_acc"].astype(float).values
    if len(s) == 0:
        return np.nan

    if not IGNORE_TRAILING_WORST_ZERO:
        return float(s[-1])

    idx = len(s) - 1
    while idx >= 0 and np.isclose(s[idx], 0.0):
        idx -= 1

    if idx < 0:
        return float(s[-1])  # all zeros
    return float(s[idx])

def run_summaries(df_run: pd.DataFrame) -> pd.Series:
    df_run = df_run.sort_values("round")

    final_round = int(df_run["round"].iloc[-1])
    final_acc = float(df_run["global_accuracy"].iloc[-1])
    best_acc = float(df_run["global_accuracy"].max())

    final_f1 = float(df_run["f1_score"].iloc[-1]) if "f1_score" in df_run.columns else np.nan
    final_auc = float(df_run["auc"].iloc[-1]) if "auc" in df_run.columns else np.nan
    final_worst = last_nonzero_worst(df_run)

    return pd.Series({
        "final_round": final_round,
        "final_accuracy": final_acc,
        "best_accuracy": best_acc,
         "final_f1": final_f1,
        "final_auc": final_auc,
        "final_worst_client_acc": final_worst,
    })

# -----------------------
#Summary table: mean ± std across seeds (per E)
# -----------------------
group_keys_run = ["algorithm", "skew_type", "local_epochs", "seed"]

per_run = (
    data.groupby(group_keys_run, group_keys=False)
    .apply(run_summaries)
    .reset_index()
)

group_keys = ["algorithm", "skew_type", "local_epochs"]

agg = per_run.groupby(group_keys).agg(
    n_seeds=("seed", "nunique"),
    final_round=("final_round", "max"),

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

agg["final_accuracy (mean±std)"] = agg.apply(
    lambda r: pm(r["final_acc_mean"], r["final_acc_std"]), axis=1
)
agg["best_accuracy (mean±std)"] = agg.apply(
    lambda r: pm(r["best_acc_mean"], r["best_acc_std"]), axis=1
)

agg["final_f1 (mean±std)"] = agg.apply(
    lambda r: pm(r["final_f1_mean"], r["final_f1_std"]), axis=1
)


agg["final_auc (mean±std)"] = agg.apply(
    lambda r: pm(r["final_auc_mean"], r["final_auc_std"]), axis=1
)
agg["final_worst_client_acc (mean±std)"] = agg.apply(
    lambda r: pm(r["final_worst_mean"], r["final_worst_std"]), axis=1
)

summary_cols = [
    "algorithm",
    "skew_type",
    "local_epochs",
    "n_seeds",
    "final_round",
    "final_accuracy (mean±std)",
    "best_accuracy (mean±std)",
     "final_f1 (mean±std)",
    "final_auc (mean±std)",
    "final_worst_client_acc (mean±std)",
]

summary = agg[summary_cols].sort_values(
    ["algorithm", "skew_type", "local_epochs"]
)

summary_path = os.path.join(OUT_DIR, "epochs_impact_summary.csv")
summary.to_csv(summary_path, index=False)

print("\n=== Saved summary table to ===")
print(summary_path)
print("\n=== Preview ===")
print(summary.head(50).to_string(index=False))

paper_table = agg[
    [
        "algorithm",
        "skew_type",
        "local_epochs",
        "final_accuracy (mean±std)",
        "best_accuracy (mean±std)",
        "final_f1 (mean±std)",
        "final_auc (mean±std)",
        "final_worst_client_acc (mean±std)",
    ]
].sort_values(["algorithm", "skew_type", "local_epochs"])

paper_table_path = os.path.join(OUT_DIR, "epochs_impact_paper_table.csv")
paper_table.to_csv(paper_table_path, index=False)

print("\nSaved paper-ready table to:")
print(paper_table_path)

print("\n=== Saved plots + tables to ===")
print(OUT_DIR)
print("\nDone.")




# Comparison plots across algorithms (for each skew type)

colors = {
    "fedavg": "#1f77b4",   
    "fedprox": "#2ca02c",  
    "fednova": "#ff7f0e",  
}

for skew in sorted(data["skew_type"].dropna().unique()):
    subset_skew = data[data["skew_type"] == skew].copy()
    if subset_skew.empty:
        continue

    fig, axes = plt.subplots(1, len(EPOCHS), figsize=(15, 5), sharey=True)

    if len(EPOCHS) == 1:
        axes = [axes]

    for ax, E in zip(axes, EPOCHS):
        sub_e = subset_skew[subset_skew["local_epochs"] == E].copy()
        if sub_e.empty:
            ax.set_visible(False)
            continue

        for algo in sorted(sub_e["algorithm"].dropna().unique()):
            sub_algo = sub_e[sub_e["algorithm"] == algo].copy()
            if sub_algo.empty:
                continue

            pivot = sub_algo.pivot_table(
                index="round",
                columns="seed",
                values="global_accuracy",
                aggfunc="mean"
            ).sort_index()

            if pivot.empty:
                continue

            mean = pivot.mean(axis=1)
            std = pivot.std(axis=1).fillna(0)

            color = colors.get(algo.lower(), None)

            ax.plot(
                mean.index,
                mean.values,
                linewidth=2.2,
                color=color,
                label=algo.upper()
            )

            ax.fill_between(
                mean.index,
                (mean - std).values,
                (mean + std).values,
                color=color,
                alpha=0.15
            )

            # final-point marker
            ax.scatter(
                mean.index[-1],
                mean.values[-1],
                color=color,
                s=28,
                zorder=3
            )

        
        ax.axhline(
            y=CENTRALIZED_ACC,
            color="black",
            linestyle="--",
            linewidth=1.8,
            label="Centralized CNN"
        )

        ax.set_title(f"E={E}", fontsize=12)
        ax.set_xlabel("Communication Round")
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Global Accuracy")

    fig.suptitle(
        f"Epoch Analysis ({skew.capitalize()} Skew, α={ALPHA}, K={K_FIXED})",
        fontsize=14,
        y=1.03
    )

   
    handles, labels = axes[0].get_legend_handles_labels()
    seen = set()
    uniq_handles = []
    uniq_labels = []
    for h, l in zip(handles, labels):
        if l not in seen:
            uniq_handles.append(h)
            uniq_labels.append(l)
            seen.add(l)

    fig.legend(
        uniq_handles,
        uniq_labels,
        loc="lower center",
        ncol=len(uniq_labels),
        bbox_to_anchor=(0.5, -0.03),
        frameon=False,
        fontsize=10
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    out_cmp = os.path.join(
        OUT_DIR,
        f"comparison_all_algorithms_{skew}_alpha{ALPHA}_K{K_FIXED}_Eimpact.png"
    )
    plt.savefig(out_cmp, dpi=200, bbox_inches="tight")
    plt.close()