"""Generate fig3_title_length.pdf — distribution of word counts in paper titles."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_loader import load_and_preprocess

OUT = os.path.join(os.path.dirname(__file__), "../report/figures/fig3_title_length.pdf")


def main():
    df, _ = load_and_preprocess()
    word_counts = df["title"].str.split().str.len()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(word_counts, bins=range(1, word_counts.max() + 2), color="#5B9BD5",
            edgecolor="white", align="left")

    ax.axvline(word_counts.median(), color="#E74C3C", linestyle="--", linewidth=1.5,
               label=f"Median = {word_counts.median():.0f} words")
    ax.legend(fontsize=9)

    ax.set_xlabel("Number of Words in Title", fontsize=10)
    ax.set_ylabel("Number of Papers", fontsize=10)
    ax.set_title("Distribution of Title Word Counts", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, 35)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    plt.savefig(OUT, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
