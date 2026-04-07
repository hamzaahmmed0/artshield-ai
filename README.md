# ArtShield AI

ArtShield AI is a real ML-powered artwork risk analysis platform for collectors, galleries, and researchers. It uses a fine-tuned DINOv2 Vision Transformer to analyze artwork attribution and return a calibrated forensic risk report.

This is not a toy classifier. It is an artist-conditioned visual anomaly detection system built on real deep learning.

---

## What It Does

A user uploads a painting and selects a claimed artist. The system:

1. Passes the image through a fine-tuned DINOv2 ViT-B/14 model
2. Extracts a 768-dimensional style embedding
3. Computes Mahalanobis distance against per-artist reference profiles
4. Returns a LOW / MEDIUM / HIGH risk signal with nearest reference works and a recommendation

Results are framed as forensic risk signals for expert review — not authenticity verdicts.

---

## Model

| Property | Detail |
|---|---|
| Base model | DINOv2 ViT-B/14 |
| Fine-tuning | Artist classification, 20 epochs |
| Embedding dim | 768 |
| Anomaly method | Mahalanobis distance + LedoitWolf covariance |
| Artists supported | 8 (WikiArt subset) |
| TPR | 80.21% |
| FPR | 31.55% |
| TNR | 68.45% |
| Top-1 Retrieval | 55.21% |

---

## Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14, Tailwind CSS, Framer Motion |
| Backend | FastAPI, Python 3.11 |
| ML | PyTorch, timm, scikit-learn |
| Model | DINOv2 ViT-B/14 (fine-tuned) |
| Dataset | WikiArt via Hugging Face |

---

## Project Structure

```text
.
├── frontend/        # Next.js web app
├── backend/         # FastAPI inference server
├── ml/              # Training, evaluation, embeddings
│   ├── data/        # Dataset download and preparation
│   ├── training/    # Fine-tuning, embedding extraction, evaluation
│   └── models/      # Saved checkpoints and artist profiles
└── docs/            # Planning notes
```

---

## Local Setup

### 1. Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

API runs at `http://127.0.0.1:8000`

### 2. Frontend

```powershell
cd frontend
npm install
npm run dev
```

Web app runs at `http://localhost:3000`

---

## ML Pipeline (if you want to retrain)

```powershell
# Download and prepare dataset
python ml/data/download_wikiart.py
python ml/data/prepare_dataset.py

# Extract baseline embeddings
python ml/training/extract_embeddings.py

# Fit artist profiles
python ml/training/fit_artist_profiles.py

# Evaluate baseline
python ml/training/evaluate_anomaly_model.py

# Fine-tune DINOv2
python ml/training/finetune_dino.py --epochs 20
```

---

## Supported Artists

- Claude Monet
- Vincent van Gogh
- Pablo Picasso
- Paul Cézanne
- Pierre-Auguste Renoir
- Rembrandt
- Salvador Dalí
- Albrecht Dürer

---

## Important Limitations

- Results are risk signals, not authenticity verdicts
- Performance varies by artist — Dalí and Van Gogh have higher false positive rates due to stylistic diversity
- The model was trained on 80 images per artist — more data will improve results
- High-quality expert forgeries may not be detected

---

## Built By

Hafiz Hamza Ahmed
© 2026 ArtShield AI
