"""Generate fig1_system_architecture.pdf — system architecture diagram."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(__file__), "../report/figures/fig1_system_architecture.pdf")


def box(ax, x, y, w, h, label, color, fontsize=9):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                          facecolor=color, edgecolor="black", linewidth=1.2)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", wrap=True)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))


fig, ax = plt.subplots(figsize=(8, 4.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")

# Input query
box(ax, 3.8, 4.8, 2.4, 0.8, "Text Query", "#AED6F1", fontsize=10)

# Split arrow down-left and down-right
arrow(ax, 4.3, 4.8, 2.0, 3.8)
arrow(ax, 5.7, 4.8, 8.0, 3.8)

# Classifier branch
box(ax, 0.5, 2.8, 3.0, 1.0, "DistilBERT\nClassifier", "#A9DFBF", fontsize=9)
arrow(ax, 2.0, 2.8, 2.0, 1.9)
box(ax, 0.5, 0.9, 3.0, 1.0, "Category Predictions\n(top-k + confidence)", "#D5F5E3", fontsize=8)

# Recommender branch
box(ax, 6.5, 2.8, 3.0, 1.0, "Sentence-Transformer\nRecommender", "#F9E79F", fontsize=9)
arrow(ax, 8.0, 2.8, 8.0, 1.9)
box(ax, 6.5, 0.9, 3.0, 1.0, "Similar Papers\n(top-k + cosine score)", "#FEF9E7", fontsize=8)

# Embedding index annotation
box(ax, 6.5, 4.0, 3.0, 0.7, "Pre-built Embedding Index", "#FAD7A0", fontsize=8)
arrow(ax, 8.0, 4.0, 8.0, 3.8)

# Combined output
arrow(ax, 2.0, 0.9, 4.3, 0.2)
arrow(ax, 8.0, 0.9, 5.7, 0.2)
box(ax, 3.5, -0.1, 3.0, 0.8, "Unified Response", "#D7BDE2", fontsize=9)

ax.set_title("System Architecture", fontsize=12, fontweight="bold", pad=10)

plt.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
plt.savefig(OUT, bbox_inches="tight")
plt.close()
print(f"Saved {OUT}")
