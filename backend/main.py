from __future__ import annotations

import logging
import time
from collections import Counter
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from config import ENV, HOST, MODEL_VERSION, PORT, SUPPORTED_ARTISTS
from ml_inference import DEVICE_NAME, MODEL_LOAD_ERROR, MODEL_READY, analyze_artwork


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
LOGGER = logging.getLogger("artshield.api")


app = FastAPI(title="ArtShield AI API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.stats = {
    "total_requests": 0,
    "risk_levels": Counter(),
    "claimed_artists": Counter(),
}

REFERENCE_ASSETS_DIR = (Path(__file__).resolve().parent / "../ml/data/processed/images_224").resolve()
REFERENCE_ASSET_INDEX = {
    file_path.name: file_path
    for file_path in REFERENCE_ASSETS_DIR.rglob("*.jpg")
}


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    LOGGER.info("%s %s completed in %.2fms", request.method, request.url.path, duration_ms)
    return response


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "ArtShield AI API", "version": "0.1.0", "docs": "/docs"}


@app.get("/api/health")
async def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model_loaded": MODEL_READY,
        "model_version": MODEL_VERSION,
        "artists_loaded": len(SUPPORTED_ARTISTS),
        "device": DEVICE_NAME,
        "environment": ENV,
        "host": HOST,
        "port": PORT,
        "error": MODEL_LOAD_ERROR,
    }


@app.get("/api/artists")
async def artists() -> dict[str, object]:
    return {"supported_artists": SUPPORTED_ARTISTS, "total": len(SUPPORTED_ARTISTS)}


@app.get("/api/stats")
async def stats() -> dict[str, object]:
    return {
        "total_requests_served": app.state.stats["total_requests"],
        "risk_level_breakdown": dict(app.state.stats["risk_levels"]),
        "claimed_artist_breakdown": dict(app.state.stats["claimed_artists"]),
    }


@app.get("/api/test-assets")
async def test_assets() -> dict[str, object]:
    sample_files = sorted(list(REFERENCE_ASSET_INDEX.keys()))[:5]
    return {"path_exists": REFERENCE_ASSETS_DIR.exists(), "sample_files": sample_files}


@app.get("/reference-assets/{filename}")
async def reference_asset(filename: str):
    asset_path = REFERENCE_ASSET_INDEX.get(filename)
    if asset_path is None or not asset_path.exists():
        raise HTTPException(status_code=404, detail=f"Reference asset '{filename}' was not found.")
    return FileResponse(asset_path)


@app.post("/api/analyze")
async def analyze(
    image: UploadFile = File(...),
    claimed_artist: str = Form(...),
) -> dict[str, object]:
    if claimed_artist not in SUPPORTED_ARTISTS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid claimed_artist '{claimed_artist}'. Supported artists: {', '.join(SUPPORTED_ARTISTS)}",
        )
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    try:
        image_bytes = await image.read()
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="The uploaded file could not be parsed as an image.") from exc

    try:
        result = analyze_artwork(pil_image, claimed_artist)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    app.state.stats["total_requests"] += 1
    app.state.stats["risk_levels"][result["risk_level"]] += 1
    app.state.stats["claimed_artists"][claimed_artist] += 1

    LOGGER.info(
        "analysis timestamp=%s claimed_artist=%s predicted_artist=%s risk_level=%s target_distance=%.4f",
        datetime.now(timezone.utc).isoformat(),
        claimed_artist,
        result["predicted_artist"],
        result["risk_level"],
        result["target_distance"],
    )
    return result
