#!/usr/bin/env python3
"""
Figure_4or9_distance.py
───────────────────────────────────────────────────────────────────────────────
Reads the CSVs produced by distance.py and recreates Figure 4

Run:
    python Figure_4or9_distance.py
"""

from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathcfg import DIST_DIR, FIG_DIR

# ────────────────────────────────
#  Global Plot Styling
# ────────────────────────────────
sns.set(style="darkgrid", context="talk", font_scale=1.6)


# ────────────────────────────────
#  I/O helpers
# ────────────────────────────────
def _distances_csv(name: str) -> Path:
    """
    Return the repo-relative path to the averaged-distance CSV for <name>.
    """
    return DIST_DIR / f"distances_{name}.csv"


def _load_and_normalise(name: str) -> pd.DataFrame:
    """
    Read a *single* averaged-distance CSV, normalise each distance value by the
    Random baseline for that (dataset, B, round), and return the processed DF.
    """
    df = pd.read_csv(_distances_csv(name), encoding="ISO-8859-1")

    random_df = df[df["method"] == "Random"]

    # Join so every row has its own Random baseline in _y columns
    merged = pd.merge(
        df,
        random_df,
        on=["dataset", "dim", "num_samples", "distance_type", "sampling_round"],
        suffixes=("_x", "_y"),
        how="left",
    )
    merged["normalised_distance"] = (
        merged["distance_value_x"] / merged["distance_value_y"]
    )

    out = merged.rename(columns={"method_x": "method"})[
        ["dataset", "dim", "num_samples",
         "distance_type", "method", "sampling_round",
         "normalised_distance"]
    ]

    desired = ["Random", "K-means", "SCCP", "Kernel_Thinning", "WCSL"]
    return out[out["method"].isin(desired)]


def load_preprocess(name: str):
    """
    Return two DataFrames: (Wasserstein, MMD) for a given experiment <name>.
    """
    df = _load_and_normalise(name)

    w_df   = df[df["distance_type"] == "W"]
    mmd_df = df[df["distance_type"] == "MMD"]
    return w_df, mmd_df


# ────────────────────────────────
#  Plotting helpers
# ────────────────────────────────
def plot_lineplot(data: pd.DataFrame,
                  title: str,
                  xlabel: str,
                  ylabel: str) -> None:
    """
    Draw a seaborn line-plot with method-specific markers, colours and dashes.
    """
    if data.empty:
        print(f"[WARN] No data found for plot: {title}")
        return

    # Method order / styles
    method_order = ["Random", "K-means", "SCCP", "Kernel_Thinning", "WCSL"]
    palette = {
        "Random": "gray",   "K-means": "blue",  "SCCP": "green",
        "Kernel_Thinning": "purple",           "WCSL": "red",
    }
    markers = {
        "Random": "o", "K-means": "s", "SCCP": "^",
        "Kernel_Thinning": "D", "WCSL": "*",
    }
    dashes = {m: (2, 2) for m in palette}
    dashes["WCSL"] = ""                       # solid line, no dash

    plt.figure(figsize=(15, 9))
    ax = sns.lineplot(
        data=data,
        x="num_samples", y="normalised_distance",
        hue="method", style="method",
        hue_order=method_order, style_order=method_order,
        palette=palette, markers=markers, dashes=dashes,
        errorbar=("ci", 95),
    )

    # Emphasise WCSL star marker
    for line in ax.lines:
        if line.get_label() == "WCSL":
            line.set_markersize(12)

    plt.title(title, fontsize=26)
    plt.xlabel(xlabel, fontsize=24)
    plt.ylabel(ylabel, fontsize=24)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.tight_layout()

    out_path = FIG_DIR / f"Fig2_{title.replace(' ', '_')}.png"
    plt.savefig(out_path)
    plt.close()
    # print(f"[✓] Saved {out_path.relative_to(Path.cwd())}")
    print(f"[✓] Saved {out_path.name}")


def plot(name: str, w_df: pd.DataFrame, mmd_df: pd.DataFrame) -> None:
    """Wrapper that draws the two distance plots for a given experiment name."""
    plot_lineplot(
        w_df,
        f"Wasserstein Distance vs. Coreset Size — {name}",
        "Coreset Size B",
        "Normalised Wasserstein Distance",
    )
    plot_lineplot(
        mmd_df,
        f"MMD vs. Coreset Size — {name}",
        "Coreset Size B",
        "Normalised MMD",
    )


# ────────────────────────────────
#  Main
# ────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_type",
        type=str,
        default="sim",
        choices=["sim", "image"],
        help='Choose between "sim" (for simulated: normal, t) or "image" (MNIST, FashionMNIST)',
    )
    args = parser.parse_args()

    # Which experiments to plot
    if args.data_type == "sim":
        flags = dict(
            MNIST=False,
            FashionMNIST=False,
            normal=True,
            t=True,
        )
    elif args.data_type == "image":
        flags = dict(
            MNIST=True,
            FashionMNIST=True,
            normal=False,
            t=False,
        )

    if flags["MNIST"]:
        w, mmd = load_preprocess("MNIST_avg")
        plot("MNIST", w, mmd)

    if flags["FashionMNIST"]:
        w, mmd = load_preprocess("FashionMNIST_avg")
        plot("FashionMNIST", w, mmd)

    if flags["normal"]:
        w, mmd = load_preprocess("normal_avg")
        plot("Normal Distribution", w, mmd)

    if flags["t"]:
        w, mmd = load_preprocess("t_avg")
        plot("t-Distribution", w, mmd)
