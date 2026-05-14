"""Generate fig7_per_class_metrics.pdf — per-class precision, recall, F1 on the test set."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_loader import load_and_preprocess, split_data, ArticleDataset
from src.classifier import load_classifier, get_device

OUT = os.path.join(os.path.dirname(__file__), "../report/figures/fig7_per_class_metrics.pdf")
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

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=range(len(label_encoder.classes_)), zero_division=0
    )

    cats = label_encoder.classes_
    order = np.argsort(f1)[::-1]
    cats_sorted = [cats[i] for i in order]
    p_sorted = precision[order]
    r_sorted = recall[order]
    f_sorted = f1[order]

    x = np.arange(len(cats_sorted))
    width = 0.26

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, p_sorted, width, label="Precision", color="#5B9BD5", edgecolor="white")
    ax.bar(x,         r_sorted, width, label="Recall",    color="#70AD47", edgecolor="white")
    ax.bar(x + width, f_sorted, width, label="F1-Score",  color="#ED7D31", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(cats_sorted, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Per-Class Precision, Recall, and F1-Score (Sorted by F1)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(0.65, color="grey", linestyle="--", linewidth=1, label="Macro F1 = 0.65")

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    plt.savefig(OUT, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
