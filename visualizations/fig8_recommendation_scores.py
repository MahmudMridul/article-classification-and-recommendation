"""Generate fig8_recommendation_scores.pdf — distribution of top-1 cosine similarity scores."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_loader import load_and_preprocess, split_data
from src.recommender import load_index, load_sbert, recommend

OUT = os.path.join(os.path.dirname(__file__), "../report/figures/fig8_recommendation_scores.pdf")
RECOMMENDER_DIR = os.path.join(os.path.dirname(__file__), "../models/recommender")


def main():
    df, _ = load_and_preprocess()
    _, _, test_df = split_data(df)

    index_data = load_index(RECOMMENDER_DIR)
    sbert = load_sbert()

    queries = test_df["title"].sample(n=200, random_state=42).tolist()

    top1_scores = []
    for q in queries:
        results = recommend(q, sbert, index_data, top_k=1)
        if results:
            top1_scores.append(results[0]["similarity"])

    top1_scores = np.array(top1_scores)
    median_score = np.median(top1_scores)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(top1_scores, bins=20, color="#5B9BD5", edgecolor="white", alpha=0.9)
    ax.axvline(median_score, color="#E74C3C", linestyle="--", linewidth=1.5,
               label=f"Median = {median_score:.2f}")

    ax.set_xlabel("Top-1 Cosine Similarity Score", fontsize=10)
    ax.set_ylabel("Number of Queries", fontsize=10)
    ax.set_title("Distribution of Top-1 Recommendation Scores\n(200 Test Queries)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    plt.savefig(OUT, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
