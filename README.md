# ArtShield AI

ArtShield AI is a beginner-friendly, publishable starter for an AI-assisted artwork risk analysis platform.

This repository is organized as a product, not just a model experiment:

- `frontend/` contains the public web app built with Next.js
- `backend/` contains the API built with FastAPI
- `ml/` contains reusable computer vision and ML modules


## What this scaffold includes

- A clean monorepo-like project structure
- A FastAPI backend with a starter `/api/v1/analyze` endpoint
- A Next.js frontend shell with an upload form and results view
- Starter ML feature extraction code for texture analysis
- Environment examples and beginner-friendly setup notes

## Product Positioning

ArtShield AI is an AI-assisted artwork risk analysis platform for collectors, galleries, and researchers.

It is **not** positioned as a final authority on art authentication.

## Project Structure

```text
.
├── backend/
├── docs/
├── frontend/
├── ml/
└── README.md
```

## Local Setup

### 1. Backend

```powershell
cd "C:\Users\PC\Documents\VS_Code\Codex P1\backend"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The API will run at `http://127.0.0.1:8000`.

### 2. Frontend

```powershell
cd "C:\Users\PC\Documents\VS_Code\Codex P1\frontend"
npm install
npm run dev
```

The web app will run at `http://localhost:3000`.

## First Milestones

1. Run the frontend and backend locally.
2. Test the upload flow end to end.
3. Replace the baseline scoring logic with real embeddings and anomaly detection.
4. Save analyses to MongoDB.
5. Add authentication and deployment.

