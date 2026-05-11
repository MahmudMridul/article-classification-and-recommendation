"""
Unified inference pipeline.
Loads the trained classifier and recommendation index once, then answers queries.
"""

import torch
from src.classifier import load_classifier, load_label_encoder, predict, get_device, CLASSIFIER_DIR
from src.recommender import load_index, load_sbert, recommend, RECOMMENDER_DIR


class Pipeline:
    def __init__(
        self,
        classifier_dir: str = CLASSIFIER_DIR,
        recommender_dir: str = RECOMMENDER_DIR,
    ):
        self.device = get_device()

        # --- Classifier ---
        self.label_encoder = load_label_encoder(classifier_dir)
        self.clf_model, self.tokenizer = load_classifier(
            classifier_dir, num_labels=len(self.label_encoder.classes_)
        )
        self.clf_model.to(self.device)
        self.clf_model.eval()

        # --- Recommender ---
        self.sbert = load_sbert(self.device)
        self.index_data = load_index(recommender_dir)

        print("[Pipeline] Ready.")

    def query(self, text: str, top_k_classify: int = 3, top_k_recommend: int = 5) -> dict:
        """
        Given a query string (keywords / paper title), return:
          - classification: top-k predicted categories with confidence
          - recommendations: top-k similar paper titles from training data
        """
        if not text or not text.strip():
            raise ValueError("Query text must not be empty.")

        classifications = predict(
            text, self.clf_model, self.tokenizer, self.label_encoder, self.device,
            top_k=top_k_classify
        )
        recommendations = recommend(text, self.sbert, self.index_data, top_k=top_k_recommend)

        return {
            "query": text,
            "classification": classifications,
            "recommendations": recommendations,
        }

    def pretty_print(self, result: dict):
        print(f"\n{'='*60}")
        print(f"Query: {result['query']}")
        print(f"{'='*60}")

        print("\n--- Classification (Top Predicted Categories) ---")
        for rank, item in enumerate(result["classification"], 1):
            print(f"  {rank}. {item['category']:<30}  confidence: {item['confidence']:.4f}")

        print("\n--- Recommendations (Most Similar Papers) ---")
        for item in result["recommendations"]:
            print(f"  [{item['rank']}] (sim={item['similarity']:.4f})  {item['title']}")
        print()
