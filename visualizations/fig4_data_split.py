"""Generate fig4_data_split.pdf — dataset split proportions pie chart."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "../report/figures/fig4_data_split.pdf")


def main():
    sizes = [28391, 3786, 5678]
    labels = ["Train\n75%\n(28,391)", "Validation\n10%\n(3,786)", "Test\n15%\n(5,678)"]
    colors = ["#5B9BD5", "#70AD47", "#ED7D31"]
    explode = (0.03, 0.03, 0.03)

    fig, ax = plt.subplots(figsize=(5, 5))
    wedges, texts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                           startangle=90, textprops={"fontsize": 10},
                           wedgeprops={"edgecolor": "white", "linewidth": 2})

    ax.set_title("Dataset Split (Stratified by Class)", fontsize=11, fontweight="bold", pad=12)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    plt.savefig(OUT, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
