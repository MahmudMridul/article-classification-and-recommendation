"""Generate fig4_pipeline_diagram.pdf — data preprocessing and training pipeline."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

OUT = os.path.join(os.path.dirname(__file__), "../report/figures/fig4_pipeline_diagram.pdf")

COLORS = {
    "data":   "#AED6F1",
    "proc":   "#A9DFBF",
    "model":  "#F9E79F",
    "output": "#D7BDE2",
}


def box(ax, cx, cy, w, h, text, color, fontsize=8):
    rect = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                          boxstyle="round,pad=0.03",
                          facecolor=color, edgecolor="#555", linewidth=1.1)
    ax.add_patch(rect)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", multialignment="center")


def arr(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.3))


fig, ax = plt.subplots(figsize=(12, 4.5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 5)
ax.axis("off")
ax.set_title("Data Preprocessing and Model Training Pipeline",
             fontsize=12, fontweight="bold", pad=10)

# Row of boxes
steps = [
    (1.0,  2.5, 1.6, 1.0, "arXiv JSON\nSnapshot",        COLORS["data"]),
    (2.8,  2.5, 1.6, 1.0, "JSON → Parquet\nConversion",  COLORS["proc"]),
    (4.6,  2.5, 1.6, 1.0, "Top-20 Category\nFiltering",  COLORS["proc"]),
    (6.4,  2.5, 1.6, 1.0, "Stratified\nSampling",        COLORS["proc"]),
    (8.2,  2.5, 1.6, 1.0, "Quality Filtering\n& Cleaning", COLORS["proc"]),
    (10.0, 2.5, 1.6, 1.0, "Train / Val / Test\nSplit",   COLORS["proc"]),
]

for cx, cy, w, h, text, color in steps:
    box(ax, cx, cy, w, h, text, color)

for i in range(len(steps) - 1):
    x1 = steps[i][0] + steps[i][2] / 2
    x2 = steps[i + 1][0] - steps[i + 1][2] / 2
    arr(ax, x1, 2.5, x2, 2.5)

# Branch from "Train / Val / Test Split" down to two models
split_cx = steps[-1][0]
arr(ax, split_cx - 0.4, 2.0, 3.5, 1.2)
arr(ax, split_cx + 0.4, 2.0, 8.5, 1.2)

box(ax, 3.5, 0.7, 3.2, 0.9, "DistilBERT Fine-tuning\n(Classifier)", COLORS["model"])
box(ax, 8.5, 0.7, 3.2, 0.9, "Sentence-Transformer\nIndex Construction", COLORS["model"])

plt.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
plt.savefig(OUT, bbox_inches="tight")
plt.close()
print(f"Saved {OUT}")


if __name__ == "__main__":
    pass  # runs at import since figure is generated at module level
