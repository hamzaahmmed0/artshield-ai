# MVP Plan

## Target Users

- New collectors who want a first-pass artwork risk screen
- Galleries that want a simple triage tool
- Researchers exploring style consistency across artworks

## MVP Goals

- Upload artwork image and metadata
- Generate a baseline risk report
- Show suspicious regions
- Explain the result in plain language
- Keep a clean architecture that can grow into a real product

## What We Delay

- Payments
- Team workspaces
- Expert review marketplace
- Marketplace scraping
- Provenance OCR
- Final authenticity claims

## Backend Responsibilities

- Accept uploads and metadata
- Run baseline image analysis
- Return a structured report
- Later: persist analyses to MongoDB

## Frontend Responsibilities

- Guide the user through upload
- Render report cards and confidence signals
- Later: account area, saved cases, and PDF exports

