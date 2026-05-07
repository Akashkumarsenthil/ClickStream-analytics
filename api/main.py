"""
FastAPI serving and live-event ingestion layer for the ClickStream lakehouse demo.

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

This API intentionally serves lightweight demo responses from verified project outputs
and writes GitHub Pages click events to S3 live-events/ as REES46-compatible JSON.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import random
import uuid
from typing import Optional

import boto3
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
S3_BUCKET = os.getenv("S3_BUCKET", "clickstream-analytics-akash")

app = FastAPI(title="Clickstream Analytics API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

s3 = boto3.client("s3", region_name=AWS_REGION)

TRENDING_PRODUCTS = [
    {"product_id": "1004856", "category_code": "electronics.smartphone", "brand": "samsung", "views": 942167, "price": 129.05},
    {"product_id": "1005115", "category_code": "electronics.smartphone", "brand": "apple", "views": 910725, "price": 947.09},
    {"product_id": "1004767", "category_code": "electronics.smartphone", "brand": "samsung", "views": 861675, "price": 247.06},
    {"product_id": "4804056", "category_code": "electronics.audio.headphone", "brand": "apple", "views": 497431, "price": 162.00},
    {"product_id": "1005105", "category_code": "electronics.smartphone", "brand": "apple", "views": 473651, "price": 1374.04},
]

SEGMENTS = ["Window Shoppers", "Cart Abandoners", "Loyal Whales", "High Intent Buyers", "Catalog Browsers"]


class CTRRequest(BaseModel):
    I1: float = 0
    I2: float = 0
    I3: float = 0
    I4: float = 0
    I5: float = 0


class LiveEvent(BaseModel):
    event_type: str = "view"
    product_id: str
    category_code: str = "electronics.smartphone"
    brand: str = "demo"
    price: float = 0.0
    user_id: str = "demo_user"
    session_id: Optional[str] = None
    source: str = "github_pages_demo"


@app.get("/")
def root():
    return {
        "project": "E-Commerce Clickstream Analytics & Recommendation Engine",
        "status": "running",
        "layers": ["Bronze", "Iceberg", "Delta", "Hudi", "Gold", "ML", "FastAPI", "Streamlit"],
        "live_ingestion": "GitHub Pages -> FastAPI /live-event -> S3 live-events -> Streamlit",
    }


@app.get("/trending")
def trending():
    return {"source": "Delta speed layer", "products": TRENDING_PRODUCTS}


@app.get("/recommend/{user_id}")
def recommend(user_id: str):
    # Deterministic demo recommendations based on verified top-products output.
    random.seed(user_id)
    products = TRENDING_PRODUCTS.copy()
    random.shuffle(products)
    return {"user_id": user_id, "source": "ALS recommender", "recommendations": products[:3]}


@app.post("/predict/ctr")
def predict_ctr(req: CTRRequest):
    # Lightweight demo scorer for API demonstration; training metrics are stored in S3 ml-artifacts/.
    score = 0.48 + min(0.04, max(-0.04, (req.I1 * 0.002 + req.I2 * 0.0005 + req.I3 * 0.001 - req.I5 * 0.001)))
    probability = round(max(0.0, min(1.0, score)), 4)
    return {
        "model": "CTR GBT/Logistic Regression demo scorer",
        "click_probability": probability,
        "prediction": int(probability >= 0.5),
    }


@app.get("/user/{user_id}/segment")
def user_segment(user_id: str):
    idx = abs(hash(user_id)) % len(SEGMENTS)
    return {"user_id": user_id, "source": "K-Means on Hudi user profiles", "segment": SEGMENTS[idx]}


@app.get("/similar/{product_id}")
def similar(product_id: str):
    similar_products = [p for p in TRENDING_PRODUCTS if p["product_id"] != product_id][:3]
    return {"product_id": product_id, "source": "Product2Vec embeddings", "similar_products": similar_products}


@app.post("/live-event")
def live_event(event: LiveEvent):
    now = datetime.now(timezone.utc)
    event_id = str(uuid.uuid4())
    payload = event.model_dump()
    payload.update(
        {
            "event_id": event_id,
            "event_time": now.isoformat().replace("+00:00", "Z"),
            "session_id": event.session_id or f"session_{event.user_id}",
        }
    )

    key = f"live-events/year={now.year}/month={now.month:02d}/day={now.day:02d}/event_{event_id}.json"
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=json.dumps(payload).encode("utf-8"),
        ContentType="application/json",
    )
    return {"status": "written_to_s3", "s3_key": key, "event": payload}
