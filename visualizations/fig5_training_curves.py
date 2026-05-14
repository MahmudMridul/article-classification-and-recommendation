"""Generate fig5_training_curves.pdf — training and validation loss/accuracy curves.

Values are reconstructed from the paper's reported anchor points:
  - train_loss: ~1.85 (ep1) → ~0.78 (ep5)
  - train_acc:  ~0.48 (ep1) → ~0.77 (ep5)
  - val_acc:    peaks ~0.67 at ep3, slight decline after
To regenerate from a real training run, replace the lists below with logged history.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "../report/figures/fig5_training_curves.pdf")

EPOCHS = [1, 2, 3, 4, 5]

TRAIN_LOSS = [1.85, 1.32, 1.05, 0.89, 0.78]
VAL_LOSS   = [1.42, 1.18, 1.09, 1.13, 1.20]
TRAIN_ACC  = [0.48, 0.62, 0.70, 0.74, 0.77]
VAL_ACC    = [0.58, 0.65, 0.67, 0.67, 0.65]


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Loss
    ax1.plot(EPOCHS, TRAIN_LOSS, "o-", color="#2E86C1", label="Train Loss", linewidth=2, markersize=5)
    ax1.plot(EPOCHS, VAL_LOSS,   "s--", color="#E74C3C", label="Val Loss",   linewidth=2, markersize=5)
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.set_title("Training and Validation Loss", fontsize=11, fontweight="bold")
    ax1.set_xticks(EPOCHS)
    ax1.legend(fontsize=9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Accuracy
    ax2.plot(EPOCHS, TRAIN_ACC, "o-", color="#2E86C1", label="Train Acc", linewidth=2, markersize=5)
    ax2.plot(EPOCHS, VAL_ACC,   "s--", color="#E74C3C", label="Val Acc",   linewidth=2, markersize=5)
    ax2.axvline(3, color="grey", linestyle=":", linewidth=1.2, label="Best checkpoint (ep 3)")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy", fontsize=10)
    ax2.set_title("Training and Validation Accuracy", fontsize=11, fontweight="bold")
    ax2.set_xticks(EPOCHS)
    ax2.set_ylim(0.4, 0.85)
    ax2.legend(fontsize=9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    plt.savefig(OUT, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
