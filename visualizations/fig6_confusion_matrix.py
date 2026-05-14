"""Generate fig6_confusion_matrix.pdf — confusion matrix on the test set."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_loader import load_and_preprocess, split_data, ArticleDataset
from src.classifier import load_classifier, get_device

OUT = os.path.join(os.path.dirname(__file__), "../report/figures/fig6_confusion_matrix.pdf")
CLASSIFIER_DIR = os.path.join(os.path.dirname(__file__), "../models/classifier")


def main():
    df, label_encoder = load_and_preprocess()
    _, _, test_df = split_data(df)

    device = get_device()
    model, tokenizer = load_classifier(CLASSIFIER_DIR, num_labels=len(label_encoder.classes_))
    model.to(device)
    model.eval()

    dataset = ArticleDataset(test_df["title"].tolist(), test_df["label"].tolist(),
                             tokenizer, max_length=128)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device))
            all_preds.extend(out.logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(batch["labels"].numpy())

    cm = confusion_matrix(all_labels, all_preds)
    # Normalize by row (true label) for better readability
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    cats = label_encoder.classes_
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    ax.set_xticks(range(len(cats)))
    ax.set_yticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(cats, fontsize=7)
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)
    ax.set_title("Confusion Matrix (Normalized by True Label)", fontsize=12, fontweight="bold")

    thresh = 0.5
    for i in range(len(cats)):
        for j in range(len(cats)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    fontsize=5, color="white" if cm_norm[i, j] > thresh else "black")

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    plt.savefig(OUT, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
