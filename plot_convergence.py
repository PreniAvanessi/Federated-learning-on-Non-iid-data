# source/plot_convergence.py
import argparse
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


def infer_algorithm_from_filename(path: str) -> str:
    name = os.path.basename(path).lower()
    if "fedprox" in name:
        return "FedProx"
    if "fednova" in name:
        return "FedNova"
    if "fedavg" in name:
        return "FedAvg"
    return "Unknown"


def load_result_csvs(patterns):
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(pat, recursive=True))
    paths = sorted(set(paths))

    if not paths:
        raise FileNotFoundError(
            "No CSV files found. Check your --csv_glob patterns."
        )

    frames = []
    for p in paths:
        df = pd.read_csv(p)

        # Normalize column names
        df.columns = [c.strip() for c in df.columns]

        # Must have round + global_accuracy
        if "round" not in df.columns or "global_accuracy" not in df.columns:
            continue

        # Add algorithm if not present
        if "algorithm" not in df.columns:
            df["algorithm"] = infer_algorithm_from_filename(p)

        # Add file path (useful for debugging)
        df["source_file"] = os.path.basename(p)

        frames.append(df)

    if not frames:
        raise ValueError(
            "Found CSV files, but none contained required columns: round, global_accuracy."
        )

    return pd.concat(frames, ignore_index=True)


def plot_convergence(
    df,
    out_dir="reports/figures",
    baseline_acc=None,
    title_prefix="CIFAR-10 Convergence",
):
    os.makedirs(out_dir, exist_ok=True)

    # Ensure needed columns exist
    for col in ["skew_type", "alpha", "algorithm"]:
        if col not in df.columns:
            df[col] = "unknown"

    # Make alpha numeric when possible
    try:
        df["alpha"] = pd.to_numeric(df["alpha"])
    except Exception:
        pass

    # We will create one figure per (skew_type) with subplots for each alpha
    skew_types = sorted(df["skew_type"].dropna().unique().tolist())

    for skew in skew_types:
        df_skew = df[df["skew_type"] == skew].copy()
        if df_skew.empty:
            continue

        alphas = sorted(df_skew["alpha"].dropna().unique().tolist())

        # Subplots: one panel per alpha (nice for side-by-side comparison)
        n = len(alphas)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
        axes = axes[0]

        for ax, a in zip(axes, alphas):
            df_a = df_skew[df_skew["alpha"] == a].copy()
            if df_a.empty:
                continue

            for algo in ["FedAvg", "FedProx", "FedNova"]:
                d = df_a[df_a["algorithm"] == algo].sort_values("round")
                if d.empty:
                    continue
                ax.plot(d["round"], d["global_accuracy"], marker="o", label=algo)

            # Optional baseline line
            if baseline_acc is not None:
                ax.axhline(float(baseline_acc), linestyle="--", label=f"Centralized ({baseline_acc:.3f})")

            ax.set_title(f"{title_prefix} | {skew} skew | alpha={a}")
            ax.set_xlabel("Federated Round")
            ax.set_ylabel("Global Accuracy")
            ax.grid(True, alpha=0.3)
            ax.legend()

        fig.tight_layout()
        out_path = os.path.join(out_dir, f"convergence_{skew}.png")
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print("Saved:", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_glob",
        nargs="+",
        default=[
            # default: looks for CSVs inside reports/**/csv
            "reports/**/csv/*.csv",
            "reports/**/csv/**/*.csv",
        ],
        help="One or more glob patterns to find result CSVs (recursive globs supported).",
    )
    parser.add_argument(
        "--baseline_acc",
        type=float,
        default=None,
        help="Optional centralized baseline accuracy (e.g., 0.755). Draws a dashed line.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="reports/figures",
        help="Where to save PNG figures.",
    )
    args = parser.parse_args()

    df = load_result_csvs(args.csv_glob)

    # Keep only the three algorithms we care about (if present)
    df = df[df["algorithm"].isin(["FedAvg", "FedProx", "FedNova"])]

    if df.empty:
        raise ValueError(
            "After filtering, no rows left for algorithms FedAvg/FedProx/FedNova. "
            "Make sure your CSV filenames include fedavg/fedprox/fednova OR add an 'algorithm' column."
        )

    plot_convergence(
        df,
        out_dir=args.out_dir,
        baseline_acc=args.baseline_acc,
    )


if __name__ == "__main__":
    main()
