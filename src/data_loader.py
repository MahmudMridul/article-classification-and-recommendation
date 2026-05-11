import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import torch


DATA_PATH = "data/data_top_20_sampled.parquet"
TOP_N_CATEGORIES = 20   # keep only the N most frequent primary categories


def load_and_preprocess(data_path: str = DATA_PATH, top_n: int = TOP_N_CATEGORIES):
    """Load parquet, extract primary category, filter to top-N categories, return clean DataFrame."""
    print(f"[DataLoader] Loading data from {data_path} ...")
    df = pd.read_parquet(data_path)
    print(f"[DataLoader] Loaded {len(df)} rows with columns: {df.columns.tolist()}")

    # Use primary (first) category only
    df["primary_category"] = df["categories"].str.split().str[0]

    # Drop rows with missing title or category
    before = len(df)
    df = df.dropna(subset=["title", "primary_category"])
    df = df[df["title"].str.strip() != ""]
    print(f"[DataLoader] Dropped {before - len(df)} rows with missing title/category. Remaining: {len(df)}")

    # Keep only top-N categories by frequency (matches 'top_20_sampled' naming)
    top_cats = df["primary_category"].value_counts().nlargest(top_n).index.tolist()
    before = len(df)
    df = df[df["primary_category"].isin(top_cats)].copy()
    print(f"[DataLoader] Filtered to top {top_n} categories. Dropped {before - len(df)} rows. Remaining: {len(df)}")

    # Encode labels
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["primary_category"])

    num_classes = len(label_encoder.classes_)
    print(f"[DataLoader] Number of unique primary categories: {num_classes}")
    print(f"[DataLoader] Categories: {list(label_encoder.classes_)}")

    return df, label_encoder


def split_data(df: pd.DataFrame, test_size: float = 0.15, val_size: float = 0.10, random_state: int = 42):
    """Split into train / val / test sets stratified by label."""
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=random_state
    )
    # val_size is fraction of remaining train_val
    adjusted_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=adjusted_val, stratify=train_val["label"], random_state=random_state
    )
    print(f"[DataLoader] Split — train: {len(train)}, val: {len(val)}, test: {len(test)}")
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


class ArticleDataset(Dataset):
    """PyTorch dataset wrapping tokenized titles and integer labels."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int = 128):
        self.labels = labels
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
