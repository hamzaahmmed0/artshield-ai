from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - fallback for partially installed envs
    def load_dotenv(*_args, **_kwargs):
        return False


BACKEND_DIR = Path(__file__).resolve().parent
load_dotenv(BACKEND_DIR / ".env")

ENV = os.getenv("ENV", "development")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

ML_MODELS_PATH = (BACKEND_DIR / os.getenv("ML_MODELS_PATH", "../ml/models")).resolve()
ML_ARTIFACTS_PATH = (BACKEND_DIR / os.getenv("ML_ARTIFACTS_PATH", "../ml/models/artifacts")).resolve()

FINETUNED_CHECKPOINT_PATH = ML_MODELS_PATH / "finetuned_dino.pth"
FINETUNED_PROFILES_PATH = ML_MODELS_PATH / "artist_profiles_finetuned.pkl"

FINETUNED_TRAIN_EMBEDDINGS_PATH = ML_ARTIFACTS_PATH / "embeddings_train_finetuned.npy"
FINETUNED_TRAIN_METADATA_PATH = ML_ARTIFACTS_PATH / "metadata_train_finetuned.json"
FINETUNED_VAL_EMBEDDINGS_PATH = ML_ARTIFACTS_PATH / "embeddings_val_finetuned.npy"
FINETUNED_VAL_LABELS_PATH = ML_ARTIFACTS_PATH / "labels_val_finetuned.npy"
FINETUNED_ARTIST_INDEX_PATH = ML_ARTIFACTS_PATH / "artist_to_index_finetuned.json"

SUPPORTED_ARTISTS = [
    "claude-monet",
    "vincent-van-gogh",
    "pablo-picasso",
    "paul-cezanne",
    "pierre-auguste-renoir",
    "rembrandt",
    "salvador-dali",
    "albrecht-durer",
]

MODEL_VERSION = "finetuned-dinov2-vitb14-20epochs"
