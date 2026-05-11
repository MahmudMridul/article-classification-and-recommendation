"""
Training entry point.

Usage:
    python train.py [--epochs N] [--batch-size N] [--lr FLOAT] [--data PATH]

Trains the DistilBERT classifier and builds the sentence-transformer
recommendation index from the training split.
"""

import argparse
import sys

from src.data_loader import load_and_preprocess, split_data
from src.classifier import train_classifier
from src.recommender import build_index


def parse_args():
    parser = argparse.ArgumentParser(description="Train classifier + build recommender index")
    parser.add_argument("--data", type=str, default="data/data_top_20_sampled.parquet",
                        help="Path to the parquet data file")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs for the classifier (default: 5)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Training batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate (default: 2e-5)")
    parser.add_argument("--max-length", type=int, default=128,
                        help="Max token length for classifier (default: 128)")
    parser.add_argument("--classifier-dir", type=str, default="models/classifier",
                        help="Directory to save the classifier")
    parser.add_argument("--recommender-dir", type=str, default="models/recommender",
                        help="Directory to save the recommender index")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  Article Classification & Recommendation — Training")
    print("=" * 60)

    # 1. Load & preprocess
    df, label_encoder = load_and_preprocess(args.data)

    # 2. Split
    train_df, val_df, test_df = split_data(df)

    # 3. Train classifier
    print("\n" + "=" * 60)
    print("  Phase 1: Training DistilBERT Classifier")
    print("=" * 60)
    train_classifier(
        train_df=train_df,
        val_df=val_df,
        label_encoder=label_encoder,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        save_dir=args.classifier_dir,
    )

    # 4. Build recommender index (on training split only — no leakage)
    print("\n" + "=" * 60)
    print("  Phase 2: Building Recommendation Index")
    print("=" * 60)
    build_index(train_df, save_dir=args.recommender_dir)

    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Classifier saved to : {args.classifier_dir}")
    print(f"  Recommender index   : {args.recommender_dir}")
    print("  Run  python test.py  to evaluate and query the pipeline.")
    print("=" * 60)


if __name__ == "__main__":
    main()
