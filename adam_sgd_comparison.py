import pandas as pd
import glob
import os

folders = {
    "adam": "reports/cifar10/csv/*.csv",
    "sgd": "reports/cifar10/csv/sgd/*.csv"
}

rows = []

for optimizer, path in folders.items():
    files = glob.glob(path)

    for file in files:
        df = pd.read_csv(file)

        # detect accuracy column
        if "global_accuracy" in df.columns:
            acc_col = "global_accuracy"
        elif "accuracy" in df.columns:
            acc_col = "accuracy"
        else:
            print(f"Skipping {file}, no accuracy column found")
            continue

        final_acc = df[acc_col].iloc[-1]
        best_acc = df[acc_col].max()

        final_f1 = df["f1_score"].iloc[-1] if "f1_score" in df.columns else None
        final_auc = df["auc"].iloc[-1] if "auc" in df.columns else None

        algorithm = os.path.basename(file).split("_")[0]

        rows.append({
            "algorithm": algorithm,
            "optimizer": optimizer,
            "final_accuracy": final_acc,
            "best_accuracy": best_acc,
            "final_f1": final_f1,
            "final_auc": final_auc
        })

results = pd.DataFrame(rows)
import os

output_path = "reports/cifar10/optimizer_comparison.csv"

results.to_csv(output_path, index=False)

print("Saved to:", os.path.abspath(output_path))

import pandas as pd

# Load your CSV
df = pd.read_csv("reports/cifar10/optimizer_comparison.csv")

# Compute averages and stability
summary = df.groupby(["algorithm", "optimizer"]).agg(
    final_accuracy_mean=("final_accuracy", "mean"),
    final_accuracy_std=("final_accuracy", "std"),
    best_accuracy_mean=("best_accuracy", "mean"),
    best_accuracy_std=("best_accuracy", "std"),
    f1_mean=("final_f1", "mean"),
    f1_std=("final_f1", "std"),
    auc_mean=("final_auc", "mean"),
    auc_std=("final_auc", "std")
).reset_index()
# Save results
summary.to_csv("reports/cifar10/optimizer_summary.csv", index=False)

print("Summary saved to reports/cifar10/optimizer_summary.csv")
print(summary)