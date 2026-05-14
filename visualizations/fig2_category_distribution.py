"""Generate fig2_category_distribution.pdf — paper count per arXiv category."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_loader import load_and_preprocess, split_data

OUT = os.path.join(os.path.dirname(__file__), "../report/figures/fig2_category_distribution.pdf")


def main():
    df, _ = load_and_preprocess()

    counts = df["primary_category"].value_counts().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(7, 7))
    bars = ax.barh(counts.index, counts.values, color="#5B9BD5", edgecolor="white", height=0.7)

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", ha="left", fontsize=8)

    ax.set_xlabel("Number of Papers", fontsize=10)
    ax.set_title("Paper Distribution Across 20 arXiv Categories", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, counts.max() * 1.12)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    plt.savefig(OUT, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
