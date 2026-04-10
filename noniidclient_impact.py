

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# Settings
# -----------------------
DATASET = "cifar10"
CSV_DIR = f"reports/{DATASET}/csv"
OUT_DIR = f"reports/{DATASET}/plots_clients_impact"
os.makedirs(OUT_DIR, exist_ok=True)

ALPHA = 0.1
LOCAL_EPOCHS = 5
CLIENTS = [10, 25, 35]
SEEDS = [43, 44, 45, 46]
SKEW_TYPES = None


all_files = glob.glob(os.path.join(CSV_DIR, "*.csv"))

dfs = []
for f in all_files:
    try:
        df = pd.read_csv(f)
    except Exception:
        continue

    required_cols = {
        "round", "global_accuracy", "num_clients",
        "skew_type", "alpha", "local_epochs",
        "seed", "worst_client_acc"
    }

    if not required_cols.issubset(df.columns):
        continue

    base = os.path.basename(f)
    algo = base.split("_", 1)[0].lower()
    df["algorithm"] = algo
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)


data = data[
    (data["alpha"].astype(float) == float(ALPHA)) &
    (data["local_epochs"].astype(int) == int(LOCAL_EPOCHS)) &
    (data["num_clients"].astype(int).isin(CLIENTS)) &
    (data["seed"].astype(int).isin(SEEDS))
].copy()

data["round"] = data["round"].astype(int)
data["num_clients"] = data["num_clients"].astype(int)
data["seed"] = data["seed"].astype(int)


group_keys_run = ["algorithm", "skew_type", "num_clients", "seed"]
NUM_ROUNDS = 30

def run_summaries(df_run):
    df_run = df_run[df_run["round"] <= NUM_ROUNDS].sort_values("round")
    return pd.Series({
        "final_accuracy": float(df_run["global_accuracy"].iloc[-1]),
        "best_accuracy": float(df_run["global_accuracy"].max()),
        "final_f1": float(df_run["f1_score"].iloc[-1]) if "f1_score" in df_run.columns else np.nan,
        "final_auc": float(df_run["auc"].iloc[-1]) if "auc" in df_run.columns else np.nan,
        "final_worst_client_acc": float(df_run["worst_client_acc"].iloc[-1]),
    })

per_run = data.groupby(group_keys_run).apply(run_summaries).reset_index()

agg = per_run.groupby(["algorithm", "skew_type", "num_clients"]).agg(
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

def pm(a, b):
    return f"{a:.4f} ± {b:.4f}"

agg["final_accuracy (mean±std)"] = agg.apply(lambda r: pm(r["final_acc_mean"], r["final_acc_std"]), axis=1)
agg["best_accuracy (mean±std)"] = agg.apply(lambda r: pm(r["best_acc_mean"], r["best_acc_std"]), axis=1)
agg["final_f1 (mean±std)"] = agg.apply(lambda r: pm(r["final_f1_mean"], r["final_f1_std"]), axis=1)
agg["final_auc (mean±std)"] = agg.apply(lambda r: pm(r["final_auc_mean"], r["final_auc_std"]), axis=1)
agg["final_worst_client_acc (mean±std)"] = agg.apply(lambda r: pm(r["final_worst_mean"], r["final_worst_std"]), axis=1)

summary = agg[
    [
        "algorithm",
        "skew_type",
        "num_clients",
        "final_accuracy (mean±std)",
        "best_accuracy (mean±std)",
        "final_f1 (mean±std)",
        "final_auc (mean±std)",
        "final_worst_client_acc (mean±std)",
    ]
].sort_values(["algorithm", "skew_type", "num_clients"])

summary_path = f"reports/{DATASET}/clients_impact_summary.csv"
summary.to_csv(summary_path, index=False)

print("Saved summary table to:")
print(summary_path)



plot_groups = data[["algorithm", "skew_type"]].drop_duplicates().values.tolist()

for algo, skew in plot_groups:
    subset = data[(data["algorithm"] == algo) & (data["skew_type"] == skew)]

    plt.figure()

    for k in CLIENTS:
        sub_k = subset[subset["num_clients"] == k]
        if sub_k.empty:
            continue

        pivot = sub_k.pivot_table(
            index="round",
            columns="seed",
            values="global_accuracy"
        ).sort_index()

        mean = pivot.mean(axis=1)
        std = pivot.std(axis=1)

        plt.plot(mean.index, mean.values, label=f"K={k}")
        plt.fill_between(mean.index,
                         (mean - std).values,
                         (mean + std).values,
                         alpha=0.2)
    centralized_acc = 0.755

    plt.axhline(
    y=centralized_acc,
    color="black",
    linestyle="--",
    linewidth=2,
    label="Centralized CNN"
   )
    plt.xlabel("Round")
    plt.ylabel("Global Accuracy")
    plt.title(f"{algo.upper()} | {skew} skew | alpha={ALPHA} | E={LOCAL_EPOCHS}")
    plt.legend()
    plt.tight_layout()

    out = os.path.join(OUT_DIR, f"{algo}_{skew}_acc_vs_round.png")
    plt.savefig(out, dpi=200)
    plt.close()

print("Saved plots to:")
print(OUT_DIR)

print("Done.")



colors = {
    "fedavg": "#1f77b4",   # blue
    "fedprox": "#2ca02c",  # green
    "fednova": "#ff7f0e",  # orange
}

for skew in sorted(data["skew_type"].dropna().unique()):
    subset_skew = data[data["skew_type"] == skew].copy()
    if subset_skew.empty:
        continue

    fig, axes = plt.subplots(1, len(CLIENTS), figsize=(15, 5), sharey=True)

    if len(CLIENTS) == 1:
        axes = [axes]

    for ax, k in zip(axes, CLIENTS):
        sub_k = subset_skew[subset_skew["num_clients"] == k].copy()
        if sub_k.empty:
            ax.set_visible(False)
            continue

        for algo in sorted(sub_k["algorithm"].dropna().unique()):
            sub_algo = sub_k[sub_k["algorithm"] == algo].copy()
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

            ax.scatter(
                mean.index[-1],
                mean.values[-1],
                color=color,
                s=28,
                zorder=3
            )

     
        ax.axhline(
            y=centralized_acc,
            color="black",
            linestyle="--",
            linewidth=1.8,
            label="Centralized CNN"
        )

        ax.set_title(f"{k} Clients", fontsize=12)
        ax.set_xlabel("Communication Round")
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Global Accuracy")

    fig.suptitle(
        f"Scalability Analysis ({skew.capitalize()} Skew, α={ALPHA}, E={LOCAL_EPOCHS})",
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
        f"comparison_all_algorithms_{skew}_alpha{ALPHA}_E{LOCAL_EPOCHS}.png"
    )
    plt.savefig(out_cmp, dpi=200, bbox_inches="tight")
    plt.close()