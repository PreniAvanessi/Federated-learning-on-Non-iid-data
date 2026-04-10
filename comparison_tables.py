import os
import glob
import pandas as pd

# ----------------------------
# Helper: find existing folder
# ----------------------------
def first_existing(paths):
    for p in paths:
        if os.path.isdir(p):
            return p
    return None

# ----------------------------
# Locate federated CSV folder
# ----------------------------
FED_CSV_DIR = first_existing([
    "reports/cifar10/csv",
    "source/reports/cifar10/csv",
])

if FED_CSV_DIR is None:
    raise FileNotFoundError(
        "Could not find federated CSV folder. Expected one of:\n"
        "- reports/cifar10/csv\n- source/reports/cifar10/csv"
    )

# Output folder
OUT_DIR = FED_CSV_DIR.replace("csv", "tables")
os.makedirs(OUT_DIR, exist_ok=True)

print("Federated CSV dir:", FED_CSV_DIR)
print("Saving tables to:", OUT_DIR)

# ----------------------------
# Load all federated CSVs
# ----------------------------
fed_files = glob.glob(os.path.join(FED_CSV_DIR, "*.csv"))
if not fed_files:
    raise FileNotFoundError(f"No CSV files found in {FED_CSV_DIR}")

fed_dfs = []
for f in fed_files:
    df = pd.read_csv(f)
    df["__file"] = os.path.basename(f)
    fed_dfs.append(df)
fed = pd.concat(fed_dfs, ignore_index=True)

# Infer algorithm from filename (works with your naming)
def infer_algo(fname: str) -> str:
    f = fname.lower()
    if "fedprox" in f:
        return "FedProx"
    if "fednova" in f:
        return "FedNova"
    if "fedavg" in f:
        return "FedAvg"
    # fallback
    return "Unknown"

if "algorithm" not in fed.columns:
    fed["algorithm"] = fed["__file"].apply(infer_algo)
else:
    # If you already have algorithm column, keep it but fill missing
    fed["algorithm"] = fed["algorithm"].fillna(fed["__file"].apply(infer_algo))

# ----------------------------
# Filter settings: CIFAR10 + 10 clients
# ----------------------------
if "dataset" in fed.columns:
    fed = fed[fed["dataset"].str.lower() == "cifar10"]

fed = fed[fed["num_clients"] == 10].copy()

# Ensure required columns exist
needed = ["round", "global_accuracy", "f1_score", "auc", "worst_client_acc", "skew_type", "alpha", "algorithm"]
missing = [c for c in needed if c not in fed.columns]
if missing:
    raise ValueError(f"Federated CSVs missing columns: {missing}")

# For each (algorithm, skew, alpha), take the last available round
last_rounds = (
    fed.groupby(["algorithm", "skew_type", "alpha"])["round"]
    .max()
    .reset_index()
    .rename(columns={"round": "last_round"})
)

fed_last = fed.merge(
    last_rounds,
    on=["algorithm", "skew_type", "alpha"],
    how="inner"
)
fed_last = fed_last[fed_last["round"] == fed_last["last_round"]].copy()

# Keep only relevant columns
fed_summary = fed_last[[
    "algorithm", "skew_type", "alpha", "round",
    "global_accuracy", "f1_score", "auc", "worst_client_acc"
]].rename(columns={"round": "fed_round"})

# ----------------------------
# Load centralized baseline CSV
# ----------------------------
BASELINE_DIR = first_existing([
    "reports/baseline/cifar10",
    "source/reports/baseline/cifar10",
])
if BASELINE_DIR is None:
    raise FileNotFoundError(
        "Could not find baseline folder. Expected one of:\n"
        "- reports/baseline/cifar10\n- source/reports/baseline/cifar10\n"
        "Make sure baseline_cifar10_log.csv exists."
    )

baseline_csv = os.path.join(BASELINE_DIR, "baseline_cifar10_log.csv")
if not os.path.isfile(baseline_csv):
    raise FileNotFoundError(f"Missing baseline file: {baseline_csv}")

base = pd.read_csv(baseline_csv)

# Choose baseline reference:
# Best test accuracy (recommended) + its metrics
best_idx = base["test_accuracy"].astype(float).idxmax()
base_best = base.loc[best_idx]

baseline_row = {
    "baseline_epoch": int(base_best["epoch"]),
    "baseline_accuracy": float(base_best["test_accuracy"]),
    "baseline_f1": float(base_best["test_f1"]),
    "baseline_auc": float(base_best["test_auc"]),
}

print("Baseline reference (best epoch):", baseline_row)

# ----------------------------
# Build comparison table
# ----------------------------
comp = fed_summary.copy()
comp["baseline_epoch"] = baseline_row["baseline_epoch"]
comp["baseline_accuracy"] = baseline_row["baseline_accuracy"]
comp["baseline_f1"] = baseline_row["baseline_f1"]
comp["baseline_auc"] = baseline_row["baseline_auc"]

# Gaps (Federated - Baseline). Negative means worse than centralized (expected).
comp["gap_accuracy"] = comp["global_accuracy"] - comp["baseline_accuracy"]
comp["gap_f1"] = comp["f1_score"] - comp["baseline_f1"]
comp["gap_auc"] = comp["auc"] - comp["baseline_auc"]

# Sort nicely
comp = comp.sort_values(["skew_type", "alpha", "algorithm"]).reset_index(drop=True)

# Save combined table
combined_path = os.path.join(OUT_DIR, "table_baseline_vs_federated_10clients.csv")
comp.to_csv(combined_path, index=False)
print("Saved:", combined_path)

# ----------------------------
# Save 4 separate tables (label/quantity × alpha)
# ----------------------------
for skew in ["label", "quantity"]:
    for alpha in [1.0, 0.3]:
        sub = comp[(comp["skew_type"] == skew) & (comp["alpha"] == alpha)].copy()
        if sub.empty:
            print(f"WARNING: no rows for skew={skew}, alpha={alpha}")
            continue
        out = os.path.join(OUT_DIR, f"table_{skew}_alpha{alpha}_10clients.csv")
        sub.to_csv(out, index=False)
        print("Saved:", out)

print("Done.")
