import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# ---------- FIND CSV FOLDER ----------
CSV_DIRS = [
    "reports/cifar10/csv",
    "source/reports/cifar10/csv",
]

CSV_DIR = next((d for d in CSV_DIRS if os.path.isdir(d)), None)
if CSV_DIR is None:
    raise FileNotFoundError("CSV directory not found")

OUT_DIR = CSV_DIR.replace("csv", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

print("Using CSV directory:", CSV_DIR)
print("Saving figures to:", OUT_DIR)

# ---------- LOAD CSVs ----------
files = glob.glob(os.path.join(CSV_DIR, "*.csv"))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# ---------- INFER ALGORITHM ----------
def infer_algo(name):
    name = name.lower()
    if "fedprox" in name:
        return "FedProx"
    if "fednova" in name:
        return "FedNova"
    return "FedAvg"

df["algorithm"] = df.get("algorithm", df["dataset"].apply(lambda _: "FedAvg"))
if "__file" not in df.columns:
    df["__file"] = df.index.astype(str)

df["algorithm"] = df["__file"].apply(infer_algo)

# ---------- TABLE 1: FINAL ROUND ----------
final_round = df.groupby(
    ["algorithm", "skew_type", "alpha", "num_clients"]
)["round"].max().reset_index()

final_df = df.merge(
    final_round,
    on=["algorithm", "skew_type", "alpha", "num_clients", "round"],
)

table_cols = [
    "algorithm", "skew_type", "alpha", "num_clients",
    "global_accuracy", "f1_score", "auc", "worst_client_acc"
]

final_table = final_df[table_cols]
final_table.to_csv(os.path.join(OUT_DIR, "table_final_results.csv"), index=False)

print("Saved table_final_results.csv")

# ---------- PLOTTING FUNCTION ----------
def plot_metric(metric, ylabel, title, fname):
    plt.figure()
    for (alg, skew, alpha), g in df.groupby(["algorithm", "skew_type", "alpha"]):
        g = g.sort_values("round")
        label = f"{alg} | {skew} | α={alpha}"
        plt.plot(g["round"], g[metric], marker="o", label=label)

    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=200)
    plt.close()

# ---------- FIGURES ----------
plot_metric("global_accuracy", "Accuracy", "Global Accuracy vs Rounds", "accuracy_vs_round.png")
plot_metric("global_loss", "Loss", "Global Loss vs Rounds", "loss_vs_round.png")
plot_metric("worst_client_acc", "Worst-client accuracy", "Worst-client Accuracy vs Rounds", "worst_client_accuracy_vs_round.png")

print("All figures generated.")
