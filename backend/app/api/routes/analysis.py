from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.schemas.analysis import AnalysisResponse, ArtworkMetadata
from app.services.analysis import analyze_artwork

router = APIRouter(tags=["analysis"])


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    image: UploadFile = File(...),
    title: str = Form(...),
    claimed_artist: str = Form(...),
    year: str | None = Form(None),
    medium: str | None = Form(None),
    notes: str | None = Form(None),
) -> AnalysisResponse:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    metadata = ArtworkMetadata(
        title=title,
        claimed_artist=claimed_artist,
        year=year,
        medium=medium,
        notes=notes,
        filename=image.filename or "upload",
        content_type=image.content_type,
    )
    image_bytes = await image.read()
    try:
        return analyze_artwork(metadata=metadata, image_bytes=image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
