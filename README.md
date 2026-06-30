# LexiForme Table Detection Service

A Python microservice that uses Microsoft's Table Transformer models to
detect tables, columns, rows, and merged cells from document images with
pixel-accurate coordinates — replacing Gemini's guesswork for table geometry.

## What it does

1. Receives a base64-encoded image via POST /detect
2. Detects all tables in the image (table detection model)
3. For each table, detects rows, columns, and merged/spanning cells
   (structure recognition model)
4. Returns a JSON structure with:
   - Exact column boundaries and width percentages
   - Exact row boundaries and height percentages
   - A full cell grid with rowspan/colspan already resolved

This JSON becomes the "ground truth" geometry that Gemini will use
instead of visually guessing column widths and merged cells.

## Local testing

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Then test with:

```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d "{\"image_base64\": \"<your base64 image string>\"}"
```

Or visit http://localhost:8000/docs for the interactive Swagger UI
where you can upload an image directly through the browser.

## Deploying to Render

1. Push this folder to a new GitHub repository
2. On Render: New > Web Service > connect this repository
3. Render will detect `render.yaml` automatically, or set manually:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. IMPORTANT: Use at least the "Starter" plan (not Free) — the Table
   Transformer models require ~2GB+ RAM to load, which exceeds the
   free tier's 512MB limit.
5. First deploy will take 5-10 minutes (downloading model weights ~115MB each)

## Response format example

```json
{
  "image_width": 1200,
  "image_height": 1600,
  "tables": [
    {
      "bbox": [120.5, 200.3, 890.1, 650.7],
      "columns": [
        {"index": 0, "x_start": 120.5, "x_end": 165.2, "width_pct": 5.8},
        {"index": 1, "x_start": 165.2, "x_end": 380.9, "width_pct": 27.8}
      ],
      "rows": [
        {"index": 0, "y_start": 200.3, "y_end": 235.1, "height_pct": 7.7}
      ],
      "cells": [
        {"row": 0, "col": 0, "rowspan": 3, "colspan": 1, "bbox": [120.5, 200.3, 165.2, 310.0]}
      ]
    }
  ]
}
```

## Notes

- This service does NOT do OCR or translation — that stays in Gemini.
  It only provides geometry (where things are), not content (what they say).
- The free Render tier will work for testing but models take 2GB+ RAM,
  so use the $7/month Starter plan for reliable operation.
- Cold starts (after 15 min idle on free tier) take 30-60s for model
  loading on top of the usual wake-up time. The Starter plan with
  "always on" avoids this.
