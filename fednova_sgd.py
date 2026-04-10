import pandas as pd
import numpy as np
import glob
import re
import os

# =========================
# CHANGE THIS FOLDER PATH
# =========================
folder = r"C:\Users\TUF\Desktop\federated noniid project 22 march\source\reports\cifar10\csv\fednova sgd"

# =========================
# CHECK FOLDER
# =========================
print("Folder exists:", os.path.exists(folder))
if not os.path.exists(folder):
    raise FileNotFoundError(f"Folder not found: {folder}")

# Load all CSV files
files = glob.glob(os.path.join(folder, "*.csv"))
print(f"Found {len(files)} CSV files")

if len(files) == 0:
    raise FileNotFoundError("No CSV files found. Check folder path or filenames.")

all_rows = []

for f in files:
    basename = os.path.basename(f)

    # Example filename:
    # fednova_cifar10_10clients_label_alpha0.1_E5_seed43_20260403_195307.csv
    match = re.search(
        r'fednova_cifar10_(\d+)clients_(label|quantity)_alpha([\d.]+)_E(\d+)_seed(\d+)',
        basename
    )

    if not match:
        print(f"Skipping (filename pattern not matched): {basename}")
        continue

    k = int(match.group(1))
    skew = match.group(2)
    alpha = float(match.group(3))
    e = int(match.group(4))
    seed = int(match.group(5))

    df = pd.read_csv(f)

    # =========================
    # KEEP ONLY ROUNDS 1 TO 30
    # =========================
    if "round" in df.columns:
        df = df[(df["round"] >= 1) & (df["round"] <= 30)].copy()
    else:
        # fallback if round column is missing
        df = df.iloc[:30].copy()

    if df.empty:
        print(f"Skipping {basename} because no rows remain after filtering to rounds 1..30")
        continue

    # Check required columns
    required_cols = ["global_accuracy", "f1_score", "auc"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Skipping {basename} because missing columns: {missing}")
        continue

    all_rows.append({
        "file": basename,
        "K": k,
        "skew": skew,
        "alpha": alpha,
        "E": e,
        "seed": seed,

        # Final metrics = round 30
        "final_acc": df["global_accuracy"].iloc[-1],
        "final_f1": df["f1_score"].iloc[-1],
        "final_auc": df["auc"].iloc[-1],

        # Best metrics = best over rounds 1..30 only
        "best_acc": df["global_accuracy"].max(),
        "best_f1": df["f1_score"].max(),
        "best_auc": df["auc"].max(),

        # Worst client at round 30
        "worst_client": df["worst_client_acc"].iloc[-1] if "worst_client_acc" in df.columns else np.nan,
    })

# Convert to dataframe
results = pd.DataFrame(all_rows)

print(f"\nLoaded {len(results)} valid rows")
print(results.head())

if results.empty:
    raise ValueError("No valid files were loaded.")

# =========================
# AGGREGATE ACROSS SEEDS
# =========================
summary = (
    results
    .groupby(["skew", "K", "E", "alpha"])
    .agg(
        final_acc_mean=("final_acc", "mean"),
        final_acc_std=("final_acc", "std"),
        best_acc_mean=("best_acc", "mean"),
        best_acc_std=("best_acc", "std"),

        final_f1_mean=("final_f1", "mean"),
        final_f1_std=("final_f1", "std"),
        best_f1_mean=("best_f1", "mean"),
        best_f1_std=("best_f1", "std"),

        final_auc_mean=("final_auc", "mean"),
        final_auc_std=("final_auc", "std"),
        best_auc_mean=("best_auc", "mean"),
        best_auc_std=("best_auc", "std"),

        worst_mean=("worst_client", "mean"),
        worst_std=("worst_client", "std"),
    )
    .reset_index()
)

print("\n=== SUMMARY ACROSS SEEDS ===")
print(summary)

# Save raw summary
summary_path = os.path.join(folder, "fednova_sgd_summary_all_settings_round1_to_30.csv")
summary.to_csv(summary_path, index=False)
print(f"\nSaved raw summary to:\n{summary_path}")

# =========================
# FORMAT mean ± std
# =========================
def pm(mean, std):
    if pd.isna(std):
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"

summary["Final Accuracy"] = summary.apply(
    lambda x: pm(x["final_acc_mean"], x["final_acc_std"]), axis=1
)
summary["Best Accuracy"] = summary.apply(
    lambda x: pm(x["best_acc_mean"], x["best_acc_std"]), axis=1
)
summary["Final F1"] = summary.apply(
    lambda x: pm(x["final_f1_mean"], x["final_f1_std"]), axis=1
)
summary["Best F1"] = summary.apply(
    lambda x: pm(x["best_f1_mean"], x["best_f1_std"]), axis=1
)
summary["Final AUC"] = summary.apply(
    lambda x: pm(x["final_auc_mean"], x["final_auc_std"]), axis=1
)
summary["Best AUC"] = summary.apply(
    lambda x: pm(x["best_auc_mean"], x["best_auc_std"]), axis=1
)
summary["Worst Client"] = summary.apply(
    lambda x: pm(x["worst_mean"], x["worst_std"]), axis=1
)

# =========================
# SAVE PRETTY SUMMARY
# =========================
pretty_summary = summary[
    [
        "skew", "K", "E", "alpha",
        "Final Accuracy", "Best Accuracy",
        "Final F1", "Best F1",
        "Final AUC", "Best AUC",
        "Worst Client"
    ]
].copy()

pretty_summary_path = os.path.join(folder, "fednova_sgd_pretty_summary_round1_to_30.csv")
pretty_summary.to_csv(pretty_summary_path, index=False)
print(f"\nSaved pretty summary to:\n{pretty_summary_path}")

# =========================
# TABLE FOR INCREASING K
# fixed alpha=0.1, E=5
# =========================
alpha_fixed = 0.1
E_fixed = 5

table_k = pretty_summary[
    (pretty_summary["alpha"] == alpha_fixed) &
    (pretty_summary["E"] == E_fixed)
].copy()

table_k = table_k.sort_values(["skew", "K"])

table_k_path = os.path.join(folder, "table_increasing_K_pretty_round1_to_30.csv")
table_k.to_csv(table_k_path, index=False)

print(f"\nSaved increasing-K table to:\n{table_k_path}")

print("\n=== TABLE: INCREASING K ===")
print(table_k.to_string(index=False))

# =========================
# OPTIONAL: LATEX ROWS
# =========================
print("\n=== LATEX ROWS ===")
for _, row in summary[
    (summary["alpha"] == alpha_fixed) &
    (summary["E"] == E_fixed)
].sort_values(["skew", "K"]).iterrows():
    print(
        f"{row['skew']} & {int(row['K'])} & "
        f"{pm(row['final_acc_mean'], row['final_acc_std'])} & "
        f"{pm(row['best_acc_mean'], row['best_acc_std'])} & "
        f"{pm(row['final_f1_mean'], row['final_f1_std'])} & "
        f"{pm(row['final_auc_mean'], row['final_auc_std'])} \\\\"
    )