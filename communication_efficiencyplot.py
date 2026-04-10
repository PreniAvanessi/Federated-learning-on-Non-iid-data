import pandas as pd
import matplotlib.pyplot as plt
import glob
import os


csv_folder = "reports/cifar10/csv"
output_dir = "reports/cifar10/plots_clients_impact"

centralized_acc = 0.7553


alpha_value = 0.1
local_epochs = 5
num_clients = 35
skew_type = "quantity"   # change to "quantity" if needed

os.makedirs(output_dir, exist_ok=True)



files = glob.glob(os.path.join(csv_folder, "*.csv"))

all_data = []

for f in files:
    try:
        df = pd.read_csv(f)
    except Exception:
        continue

    algo = os.path.basename(f).split("_")[0].lower()
    df["algorithm"] = algo
    all_data.append(df)

data = pd.concat(all_data, ignore_index=True)



data = data[
    (data["alpha"] == alpha_value) &
    (data["local_epochs"] == local_epochs) &
    (data["num_clients"] == num_clients) &
    (data["skew_type"] == skew_type)
].copy()


data["communication_mb"] = data["total_traffic_bytes"] / 1e6
data["round_comm_mb"] = data["round_traffic_bytes"] / 1e6


plt.figure(figsize=(8,6))

algorithms = sorted(data["algorithm"].unique())

for algo in algorithms:

    df_algo = data[data["algorithm"] == algo]

    grouped = (
        df_algo.groupby("round")
        .agg(
            mean_acc=("global_accuracy", "mean"),
            std_acc=("global_accuracy", "std"),
            mean_comm=("communication_mb", "mean")
        )
        .reset_index()
    )

    plt.plot(
        grouped["mean_comm"],
        grouped["mean_acc"],
        linewidth=2,
        label=algo.upper()
    )

    plt.fill_between(
        grouped["mean_comm"],
        grouped["mean_acc"] - grouped["std_acc"],
        grouped["mean_acc"] + grouped["std_acc"],
        alpha=0.2
    )

# centralized baseline
plt.axhline(
    y=centralized_acc,
    color="black",
    linestyle="--",
    linewidth=2,
    label="Centralized CNN"
)

plt.xlabel("Total Communication (MB)")
plt.ylabel("Global Accuracy")
plt.title(
    f"Communication Efficiency ({skew_type} skew, "
    f"α={alpha_value}, K={num_clients}, E={local_epochs})"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

plot_path = os.path.join(output_dir, "communication_efficiency.png")
plt.savefig(plot_path, dpi=300)
plt.show()

print("Plot saved to:", plot_path)



rows = []

for algo in algorithms:

    df_algo = data[data["algorithm"] == algo]

    # last round per seed
    last_round = df_algo.groupby("seed").tail(1)

    final_acc = last_round["global_accuracy"].mean()
    final_acc_std = last_round["global_accuracy"].std()

    total_comm = last_round["communication_mb"].mean()
    runtime = last_round["total_runtime"].mean()

    bytes_per_round = df_algo["round_comm_mb"].mean()

    rows.append({
        "Algorithm": algo.upper(),
        "Final Accuracy": round(final_acc,4),
        "Std Accuracy": round(final_acc_std,4),
        "Total Communication (MB)": round(total_comm,2),
        "Communication per Round (MB)": round(bytes_per_round,2),
        "Runtime (s)": round(runtime,2)
    })

table = pd.DataFrame(rows)

table_path = os.path.join(output_dir, "communication_efficiency_table.csv")
table.to_csv(table_path, index=False)


plot_path = os.path.join(output_dir, f"communication_efficiency_{skew_type}.png")

plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()

print("Plot saved to:", plot_path)
table_path = os.path.join(
    output_dir,
    f"communication_efficiency_table_{skew_type}.csv"
)

table.to_csv(table_path, index=False)

print("\nCommunication Efficiency Table:")
print(table)
print("\nTable saved to:", table_path)