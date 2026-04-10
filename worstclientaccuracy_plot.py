import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# 📁 CHANGE ONLY IF NEEDED
CSV_DIR = r"C:\Users\TUF\Desktop\federated noniid project\source\reports\cifar10\csv\10clients 5 epochs"

OUT_DIR = os.path.join(CSV_DIR, "plots")
os.makedirs(OUT_DIR, exist_ok=True)

METHODS = ["fedavg", "fedprox", "fednova"]
ALPHAS = ["0.1", "0.3"]
SKEWS = ["label", "quantity"]


def find_csv(method, skew, alpha):
    pattern = os.path.join(
        CSV_DIR, f"{method}_cifar10_10clients_{skew}_alpha{alpha}*.csv"
    )
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def load_series(csv_path):
    df = pd.read_csv(csv_path)

    df = df[["round", "worst_client_acc"]].copy()
    df = df.sort_values("round")
    df = df.drop_duplicates(subset=["round"], keep="last")

    # 🔧 Drop Flower artifact last round
    if len(df) > 1:
        df = df.iloc[:-1]

    return df["round"].tolist(), df["worst_client_acc"].tolist()


def plot_worst_client_for_skew(skew):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, alpha in zip(axes, ALPHAS):
        for method in METHODS:
            csv_path = find_csv(method, skew, alpha)
            if csv_path is None:
                print(f"⚠ Missing: {method}, {skew}, alpha={alpha}")
                continue

            rounds, worst_acc = load_series(csv_path)

            ax.plot(
                rounds,
                worst_acc,
                marker="o",
                linewidth=2,
                label=method.capitalize(),
            )

        ax.set_title(f"{skew.capitalize()} Skew | alpha={alpha}")
        ax.set_xlabel("Federated Round")
        ax.grid(True)

    axes[0].set_ylabel("Worst Client Accuracy")
    axes[0].legend()

    plt.suptitle(f"CIFAR-10 Worst Client Accuracy ({skew.capitalize()} Skew)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join(
        OUT_DIR, f"worst_client_accuracy_{skew}_alpha0.1_vs_0.3.png"
    )
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("✅ Saved:", out_path)


# 🚀 Run for both skews
for skew in SKEWS:
    plot_worst_client_for_skew(skew)
