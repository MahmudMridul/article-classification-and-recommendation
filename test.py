"""
Testing / inference entry point.

Usage:
    # Evaluate on test set + run a few demo queries:
    python test.py

    # Interactive query mode:
    python test.py --interactive

    # Single query from CLI:
    python test.py --query "neural networks image recognition"

    # Evaluate on test set only (no demo queries):
    python test.py --eval-only
"""

import argparse
import sys

from src.data_loader import load_and_preprocess, split_data
from src.classifier import test_classifier, CLASSIFIER_DIR
from src.pipeline import Pipeline


DEMO_QUERIES = [
    "deep learning image classification convolutional neural network",
    "black hole accretion disk gravitational waves",
    "natural language processing transformer attention mechanism",
    "quantum computing entanglement superconducting qubits",
    "graph neural networks social network analysis",
    "galaxy formation dark matter cosmological simulation",
    "reinforcement learning policy gradient reward optimization",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Test and query the trained pipeline")
    parser.add_argument("--data", type=str, default="data/data_top_20_sampled.parquet",
                        help="Path to the parquet data file")
    parser.add_argument("--classifier-dir", type=str, default=CLASSIFIER_DIR,
                        help="Directory with the saved classifier")
    parser.add_argument("--recommender-dir", type=str, default="models/recommender",
                        help="Directory with the saved recommender index")
    parser.add_argument("--query", type=str, default=None,
                        help="Single query to classify and get recommendations for")
    parser.add_argument("--interactive", action="store_true",
                        help="Enter interactive query loop")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only run evaluation on the test set, skip demo queries")
    parser.add_argument("--top-k-classify", type=int, default=3,
                        help="Number of top categories to return (default: 3)")
    parser.add_argument("--top-k-recommend", type=int, default=5,
                        help="Number of recommendations to return (default: 5)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  Article Classification & Recommendation — Testing")
    print("=" * 60)

    # 1. Evaluate classifier on held-out test set
    if not args.query:
        print("\n[Step 1] Evaluating classifier on test set ...")
        df, label_encoder = load_and_preprocess(args.data)
        _, _, test_df = split_data(df)
        test_classifier(test_df, label_encoder, save_dir=args.classifier_dir)

    if args.eval_only:
        return

    # 2. Load unified pipeline
    print("\n[Step 2] Loading inference pipeline ...")
    pipeline = Pipeline(
        classifier_dir=args.classifier_dir,
        recommender_dir=args.recommender_dir,
    )

    # 3. Single CLI query
    if args.query:
        result = pipeline.query(
            args.query,
            top_k_classify=args.top_k_classify,
            top_k_recommend=args.top_k_recommend,
        )
        pipeline.pretty_print(result)
        return

    # 4. Demo queries
    print("\n[Step 3] Running demo queries ...")
    for q in DEMO_QUERIES:
        result = pipeline.query(
            q,
            top_k_classify=args.top_k_classify,
            top_k_recommend=args.top_k_recommend,
        )
        pipeline.pretty_print(result)

    # 5. Interactive mode
    if args.interactive:
        print("\n" + "=" * 60)
        print("  Interactive Query Mode  (type 'quit' to exit)")
        print("=" * 60)
        while True:
            try:
                text = input("\nEnter keywords or paper title: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break
            if text.lower() in ("quit", "exit", "q"):
                break
            if not text:
                continue
            try:
                result = pipeline.query(
                    text,
                    top_k_classify=args.top_k_classify,
                    top_k_recommend=args.top_k_recommend,
                )
                pipeline.pretty_print(result)
            except Exception as e:
                print(f"[Error] {e}")


if __name__ == "__main__":
    main()
