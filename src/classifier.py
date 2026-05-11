import os
import json
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

from src.data_loader import ArticleDataset

MODEL_NAME = "distilbert-base-uncased"
CLASSIFIER_DIR = "models/classifier"


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Classifier] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[Classifier] GPU not available, using CPU.")
    return device


def train_classifier(
    train_df,
    val_df,
    label_encoder,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 2e-5,
    max_length: int = 128,
    save_dir: str = CLASSIFIER_DIR,
):
    device = get_device()
    num_labels = len(label_encoder.classes_)

    print(f"[Classifier] Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
    model.to(device)

    print(f"[Classifier] Tokenizing {len(train_df)} training samples ...")
    train_dataset = ArticleDataset(train_df["title"].tolist(), train_df["label"].tolist(), tokenizer, max_length)
    val_dataset = ArticleDataset(val_df["title"].tolist(), val_df["label"].tolist(), tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps
    )

    best_val_acc = 0.0
    print(f"\n[Classifier] Starting training — {epochs} epochs, batch_size={batch_size}, lr={lr}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=True)
        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        # Validation
        val_acc, val_loss = evaluate(model, val_loader, device)

        print(
            f"[Classifier] Epoch {epoch}/{epochs} — "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"[Classifier] New best val_acc={best_val_acc:.4f} — saving checkpoint ...")
            save_classifier(model, tokenizer, label_encoder, save_dir)

    print(f"\n[Classifier] Training complete. Best val_acc={best_val_acc:.4f}")
    return model, tokenizer


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total, total_loss / len(loader)


def test_classifier(test_df, label_encoder, save_dir: str = CLASSIFIER_DIR, batch_size: int = 32, max_length: int = 128):
    """Evaluate saved classifier on the test set and print a full report."""
    device = get_device()
    model, tokenizer = load_classifier(save_dir, num_labels=len(label_encoder.classes_))
    model.to(device)
    model.eval()

    test_dataset = ArticleDataset(test_df["title"].tolist(), test_df["label"].tolist(), tokenizer, max_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    all_preds = []
    all_labels = []

    print("[Classifier] Running inference on test set ...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\n[Classifier] Test Accuracy: {acc:.4f}")
    print("\n[Classifier] Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_, zero_division=0))
    return acc


def save_classifier(model, tokenizer, label_encoder, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    with open(os.path.join(save_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)
    print(f"[Classifier] Saved to {save_dir}")


def load_classifier(save_dir: str, num_labels: int = None):
    print(f"[Classifier] Loading from {save_dir} ...")
    with open(os.path.join(save_dir, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    n = num_labels or len(label_encoder.classes_)
    tokenizer = DistilBertTokenizerFast.from_pretrained(save_dir)
    model = DistilBertForSequenceClassification.from_pretrained(save_dir, num_labels=n)
    return model, tokenizer


def load_label_encoder(save_dir: str = CLASSIFIER_DIR):
    with open(os.path.join(save_dir, "label_encoder.pkl"), "rb") as f:
        return pickle.load(f)


def predict(query: str, model, tokenizer, label_encoder, device, top_k: int = 3, max_length: int = 128):
    """Return top-k predicted categories with confidence scores for a query string."""
    model.eval()
    encoding = tokenizer(
        query,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

    top_indices = np.argsort(probs)[::-1][:top_k]
    results = [
        {"category": label_encoder.classes_[i], "confidence": float(probs[i])}
        for i in top_indices
    ]
    return results
