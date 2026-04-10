

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATASET = "cifar10"
CSV_DIR = f"reports/{DATASET}/csv"
OUT_DIR = f"reports/{DATASET}/plots_alpha_impact"
os.makedirs(OUT_DIR, exist_ok=True)

ALPHAS = [0.1, 0.3, 0.5]
K_FIXED = 35
LOCAL_EPOCHS = 5
SEEDS = [43, 44, 45, 46]
SKEW_TYPES = None
ALGORITHMS = None

NUM_ROUNDS = 30
IGNORE_TRAILING_WORST_ZERO = True
CENTRALIZED_ACC = 0.7553



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
    "worst_client_acc",
}

for f in all_files:
    try:
        df = pd.read_csv(f)
    except Exception:
        continue

    if not required_cols.issubset(df.columns):
        continue

    base = os.path.basename(f)
    algo = base.split("_", 1)[0].lower()
    df["algorithm"] = algo
    df["source_file"] = base
    dfs.append(df)

if not dfs:
    raise ValueError("No usable CSVs found.")

data = pd.concat(dfs, ignore_index=True)


data["round"] = data["round"].astype(int)
data["num_clients"] = data["num_clients"].astype(int)
data["seed"] = data["seed"].astype(int)
data["local_epochs"] = data["local_epochs"].astype(int)
data["alpha"] = data["alpha"].astype(float)

data = data[
    (data["alpha"].isin(ALPHAS)) &
    (data["num_clients"] == int(K_FIXED)) &
    (data["local_epochs"] == int(LOCAL_EPOCHS)) &
    (data["seed"].isin(SEEDS))
].copy()

if SKEW_TYPES is not None:
    data = data[data["skew_type"].isin(SKEW_TYPES)].copy()

if ALGORITHMS is not None:
    data = data[data["algorithm"].isin([a.lower() for a in ALGORITHMS])].copy()

if data.empty:
    raise ValueError(
        "After filtering, no rows remain.\n"
        f"ALPHAS={ALPHAS}, K_FIXED={K_FIXED}, LOCAL_EPOCHS={LOCAL_EPOCHS}, SEEDS={SEEDS}"
    )


def pm(a, b, digits=4):
    if pd.isna(a):
        return ""
    if pd.isna(b):
        return f"{a:.{digits}f}"
    return f"{a:.{digits}f} ± {b:.{digits}f}"

def last_nonzero_worst(df_run: pd.DataFrame) -> float:
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

group_keys_run = ["algorithm", "skew_type", "alpha", "seed"]

per_run = (
    data.groupby(group_keys_run, group_keys=False)
    .apply(run_summaries)
    .reset_index()
)

agg = per_run.groupby(["algorithm", "skew_type", "alpha"]).agg(
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

agg["final_accuracy (mean±std)"] = agg.apply(lambda r: pm(r["final_acc_mean"], r["final_acc_std"]), axis=1)
agg["best_accuracy (mean±std)"] = agg.apply(lambda r: pm(r["best_acc_mean"], r["best_acc_std"]), axis=1)
agg["final_f1 (mean±std)"] = agg.apply(lambda r: pm(r["final_f1_mean"], r["final_f1_std"]), axis=1)
agg["final_auc (mean±std)"] = agg.apply(lambda r: pm(r["final_auc_mean"], r["final_auc_std"]), axis=1)
agg["final_worst_client_acc (mean±std)"] = agg.apply(lambda r: pm(r["final_worst_mean"], r["final_worst_std"]), axis=1)

summary = agg[
    [
        "algorithm",
        "skew_type",
        "alpha",
        "final_accuracy (mean±std)",
        "best_accuracy (mean±std)",
        "final_f1 (mean±std)",
        "final_auc (mean±std)",
        "final_worst_client_acc (mean±std)",
    ]
].sort_values(["algorithm", "skew_type", "alpha"])

summary_path = os.path.join(OUT_DIR, "alpha_impact_summary.csv")
summary.to_csv(summary_path, index=False)

paper_table_path = os.path.join(OUT_DIR, "alpha_impact_paper_table.csv")
summary.to_csv(paper_table_path, index=False)

print("Saved summary table to:")
print(summary_path)



plot_groups = data[["algorithm", "skew_type"]].drop_duplicates().values.tolist()

for algo, skew in plot_groups:
    subset = data[(data["algorithm"] == algo) & (data["skew_type"] == skew)].copy()
    if subset.empty:
        continue

    plt.figure(figsize=(8, 5))

    for a in ALPHAS:
        sub_a = subset[subset["alpha"] == a].copy()
        if sub_a.empty:
            continue

        pivot = sub_a.pivot_table(
            index="round",
            columns="seed",
            values="global_accuracy",
            aggfunc="mean"
        ).sort_index()

        if pivot.empty:
            continue

        mean = pivot.mean(axis=1)
        std = pivot.std(axis=1).fillna(0)

        plt.plot(mean.index, mean.values, label=f"α={a}")
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

    plt.xlabel("Round")
    plt.ylabel("Global Accuracy")
    plt.title(f"{algo.upper()} | {skew} skew | K={K_FIXED} | E={LOCAL_EPOCHS} | Impact of α")
    plt.legend()
    plt.tight_layout()

    out = os.path.join(OUT_DIR, f"{algo}_{skew}_acc_vs_round_K{K_FIXED}_E{LOCAL_EPOCHS}_alphaimpact.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()


colors = {
    "fedavg": "#1f77b4",   # blue
    "fedprox": "#2ca02c",  # green
    "fednova": "#ff7f0e",  # orange
}

for skew in sorted(data["skew_type"].dropna().unique()):
    subset_skew = data[data["skew_type"] == skew].copy()
    if subset_skew.empty:
        continue

    fig, axes = plt.subplots(1, len(ALPHAS), figsize=(15, 5), sharey=True)

    if len(ALPHAS) == 1:
        axes = [axes]

    for ax, a in zip(axes, ALPHAS):
        sub_a = subset_skew[subset_skew["alpha"] == a].copy()
        if sub_a.empty:
            ax.set_visible(False)
            continue

        for algo in sorted(sub_a["algorithm"].dropna().unique()):
            sub_algo = sub_a[sub_a["algorithm"] == algo].copy()
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

        # Centralized baseline
        ax.axhline(
            y=CENTRALIZED_ACC,
            color="black",
            linestyle="--",
            linewidth=1.8,
            label="Centralized CNN"
        )

        ax.set_title(f"α={a}", fontsize=12)
        ax.set_xlabel("Communication Round")
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Global Accuracy")

    fig.suptitle(
        f"Heterogeneity Analysis ({skew.capitalize()} Skew, K={K_FIXED}, E={LOCAL_EPOCHS})",
        fontsize=14,
        y=1.03
    )

    # build one clean legend for the whole figure
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
        f"comparison_all_algorithms_{skew}_K{K_FIXED}_E{LOCAL_EPOCHS}_alphaimpact.png"
    )
    plt.savefig(out_cmp, dpi=200, bbox_inches="tight")
    plt.close()