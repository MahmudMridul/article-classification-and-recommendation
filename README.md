# Article Classification & Recommendation

A two-in-one ML pipeline that takes a few keywords or a paper title as input and returns:

1. **Classification** — predicted arXiv research category (top-3 with confidence scores)
2. **Recommendation** — most similar paper titles from the training corpus

---

## Project Summary

### Architecture

| Component | Model | Purpose |
|-----------|-------|---------|
| Classifier | `distilbert-base-uncased` fine-tuned | Maps keyword/title → arXiv category |
| Recommender | `all-MiniLM-L6-v2` (sentence-transformer) | Embeds titles for cosine-similarity search |

### Data

- Source: `data/data_top_20_sampled.parquet` — 47,500 arXiv papers (`id`, `title`, `abstract`, `categories`)
- Filtering: top 20 most frequent primary categories retained → **37,855 papers**
- Split: 75% train / 10% val / 15% test (stratified)

### Categories (20 classes)

`astro-ph`, `astro-ph.CO`, `astro-ph.GA`, `cond-mat.mes-hall`, `cond-mat.mtrl-sci`,
`cond-mat.stat-mech`, `cond-mat.str-el`, `cond-mat.supr-con`, `cs.AI`, `cs.CL`,
`cs.CV`, `cs.LG`, `gr-qc`, `hep-ph`, `hep-th`, `math-ph`, `math.AP`,
`math.CO`, `quant-ph`, `stat.ML`

### Performance (test set)

| Metric | Score |
|--------|-------|
| Test Accuracy | 67.65% |
| Macro F1 | 0.65 |
| Weighted F1 | 0.67 |

Notable per-class F1: `math.CO` 0.86, `cs.CV` 0.83, `math.AP` 0.80, `cs.CL` 0.84

### Project Structure

```
article_classification_recommendation/
├── data/
│   └── data_top_20_sampled.parquet
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # load, preprocess, split, PyTorch Dataset
│   ├── classifier.py       # DistilBERT train / eval / predict
│   ├── recommender.py      # sentence-transformer index build / search
│   └── pipeline.py         # unified inference (classify + recommend)
├── models/
│   ├── classifier/         # saved DistilBERT weights + tokenizer + label encoder
│   └── recommender/        # saved embedding index (pickle)
├── train.py                # training entry point
├── test.py                 # evaluation & query entry point
└── README.md
```

---

## Setup

### Requirements

- Python 3.12
- NVIDIA GPU with CUDA 12.4 (CPU fallback is automatic if no GPU is available)

### Install

```bash
# Create and activate the virtual environment
uv venv --python 3.12 .venv
source .venv/bin/activate

# Install PyTorch with CUDA 12.4
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install remaining dependencies
uv pip install transformers sentence-transformers scikit-learn numpy pandas pyarrow tqdm accelerate
```

---

## Training

Run with defaults (5 epochs, batch size 32, lr 2e-5):

```bash
source .venv/bin/activate
python train.py
```

All available options:

```bash
python train.py \
  --epochs 5 \
  --batch-size 32 \
  --lr 2e-5 \
  --max-length 128 \
  --data data/data_top_20_sampled.parquet \
  --classifier-dir models/classifier \
  --recommender-dir models/recommender
```

Training runs two phases:
1. **Phase 1** — Fine-tunes DistilBERT on paper titles (best checkpoint saved by val accuracy)
2. **Phase 2** — Encodes all training titles with a sentence-transformer and saves the index

Expected training time on RTX 3060: ~15 minutes for 5 epochs.

---

## Testing

### Evaluate on the test set + run demo queries

```bash
source .venv/bin/activate
python test.py
```

### Single query from the command line

```bash
python test.py --query "black hole gravitational waves merger"
```

### Interactive mode

```bash
python test.py --interactive
```

In interactive mode, type any keywords or paper title and press Enter. Type `quit` to exit.

### Evaluation only (no queries)

```bash
python test.py --eval-only
```

### Tune output size

```bash
python test.py --query "deep learning object detection" \
               --top-k-classify 5 \
               --top-k-recommend 10
```

---

## Example Output

```
Query: deep learning image classification convolutional neural network

--- Classification (Top Predicted Categories) ---
  1. cs.CV                           confidence: 0.9850
  2. cs.LG                           confidence: 0.0067
  3. stat.ML                         confidence: 0.0023

--- Recommendations (Most Similar Papers) ---
  [1] (sim=0.6708)  Deep Convolutional Decision Jungle for Image Classification
  [2] (sim=0.6313)  Semi-Supervised Deep Learning for Fully Convolutional Networks
  [3] (sim=0.5920)  Deep Versus Wide Convolutional Neural Networks for Object Recognition on Neuromorphic System
  [4] (sim=0.5837)  Deep Predictive Coding Network for Object Recognition
  [5] (sim=0.5712)  Learning Transferrable Knowledge for Semantic Segmentation with Deep Convolutional Neural Network
```
