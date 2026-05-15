import os
import pickle
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

RECOMMENDER_DIR = "models/recommender"
SBERT_MODEL = "all-MiniLM-L6-v2"


def get_device():
    if torch.cuda.is_available():
        print(f"[Recommender] Using GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"
    print("[Recommender] GPU not available, using CPU.")
    return "cpu"


def build_index(train_df, save_dir: str = RECOMMENDER_DIR, batch_size: int = 256):
    """
    Encode all training paper titles with a sentence-transformer and save
    the embedding matrix + title list for similarity search.
    """
    device = get_device()
    print(f"[Recommender] Loading sentence-transformer model: {SBERT_MODEL}")
    model = SentenceTransformer(SBERT_MODEL, device=device)

    titles = train_df["title"].tolist()
    ids = train_df["id"].tolist()
    print(f"[Recommender] Encoding {len(titles)} paper titles in batches of {batch_size} ...")

    embeddings = model.encode(
        titles,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print(f"[Recommender] Embeddings shape: {embeddings.shape}")

    os.makedirs(save_dir, exist_ok=True)
    index_data = {"titles": titles, "ids": ids, "embeddings": embeddings}
    with open(os.path.join(save_dir, "index.pkl"), "wb") as f:
        pickle.dump(index_data, f)
    print(f"[Recommender] Index saved to {save_dir}/index.pkl")
    return model, index_data


def load_index(save_dir: str = RECOMMENDER_DIR):
    index_path = os.path.join(save_dir, "index.pkl")
    print(f"[Recommender] Loading index from {index_path} ...")
    with open(index_path, "rb") as f:
        index_data = pickle.load(f)
    print(f"[Recommender] Index loaded: {len(index_data['titles'])} papers")
    return index_data


def load_sbert(device: str = None):
    if device is None:
        device = get_device()
    print(f"[Recommender] Loading sentence-transformer: {SBERT_MODEL}")
    return SentenceTransformer(SBERT_MODEL, device=device, local_files_only=True)


def recommend(query: str, model, index_data: dict, top_k: int = 5):
    """
    Given a query string, return top-k most similar paper titles using
    cosine similarity against the pre-built embedding index.
    """
    query_emb = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # Cosine similarity — embeddings are L2-normalised so dot product = cosine sim
    scores = (index_data["embeddings"] @ query_emb.T).squeeze()
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = [
        {
            "rank": rank + 1,
            "title": index_data["titles"][i],
            "id": index_data["ids"][i],
            "similarity": float(scores[i]),
        }
        for rank, i in enumerate(top_indices)
    ]
    return results
