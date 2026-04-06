# ML Module

This folder now contains the real local machine-learning pipeline for ArtShield AI.

## What Runs Locally

- Python packages install into your local environment
- WikiArt downloads onto your machine
- DINOv2 weights download onto your machine
- Embeddings, artist profiles, plots, and evaluation outputs are saved inside this project

## Current Structure

```text
ml/
├── data/
│   ├── download_wikiart.py
│   ├── prepare_dataset.py
│   ├── processed/
│   └── README.md
├── explainability/
│   └── regions.py
├── features/
│   └── texture.py
├── models/
│   ├── artifacts/
│   └── risk_baseline.py
├── training/
│   ├── extract_embeddings.py
│   ├── fit_artist_profiles.py
│   ├── evaluate_anomaly_model.py
│   ├── finetune_dino.py
│   ├── run_inference.py
│   └── utils.py
├── requirements.txt
└── README.md
```

## Real Pipeline Order

1. `data/download_wikiart.py`
2. `data/prepare_dataset.py`
3. `training/extract_embeddings.py`
4. `training/fit_artist_profiles.py`
5. `training/evaluate_anomaly_model.py`
6. `training/run_inference.py`
7. `training/finetune_dino.py`

## Why This Is The Right First Version

We are using a pretrained frozen DINOv2 encoder as a real feature extractor, then building per-artist anomaly profiles on top of those embeddings. This is not a fake demo model, and it is a strong production-minded baseline before fine-tuning.
