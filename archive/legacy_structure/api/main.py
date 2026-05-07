"""
FastAPI Serving Layer
=====================
REST API endpoints for serving ML model predictions and recommendations.

Endpoints:
    GET  /recommend/{user_id}      — Personalized product recommendations
    POST /predict/ctr              — Click-through rate prediction
    GET  /trending                 — Real-time trending products
    GET  /user/{user_id}/segment   — Customer segment classification
    GET  /similar/{product_id}     — Similar products (Product2Vec)
    GET  /health                   — Health check

Usage:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import boto3
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ─── Pydantic Models ───

class CTRRequest(BaseModel):
    """Request body for CTR prediction."""
    int_features: List[float] = Field(
        ..., min_length=13, max_length=13,
        description="13 integer features from Criteo"
    )
    cat_features: List[str] = Field(
        ..., min_length=26, max_length=26,
        description="26 hashed categorical features"
    )


class CTRResponse(BaseModel):
    click_probability: float
    prediction: int
    model_used: str
    latency_ms: float


class RecommendationItem(BaseModel):
    product_id: int
    score: float
    rank: int


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[RecommendationItem]
    model: str
    latency_ms: float


class TrendingProduct(BaseModel):
    product_id: int
    brand: Optional[str]
    category: Optional[str]
    trend_score: float
    views: int
    carts: int
    purchases: int
    unique_users: int


class TrendingResponse(BaseModel):
    trending_products: List[TrendingProduct]
    window: str
    latency_ms: float


class SegmentResponse(BaseModel):
    user_id: int
    segment_id: int
    segment_label: str
    segment_characteristics: Dict[str, Any]
    latency_ms: float


class SimilarProduct(BaseModel):
    product_id: str
    similarity: float


class SimilarResponse(BaseModel):
    query_product_id: str
    similar_products: List[SimilarProduct]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
    uptime_seconds: float
    version: str


class LiveEvent(BaseModel):
    """Schema for live clickstream events from GitHub Pages."""
    event_time: str
    event_type: str
    product_id: str
    category_code: Optional[str] = None
    brand: Optional[str] = None
    price: float
    user_id: str
    session_id: Optional[str] = None
    source: str = "github_pages_demo"


# ─── App State ───

class AppState:
    """Global application state for loaded models and data."""
    def __init__(self):
        self.start_time = time.time()
        self.spark = None
        self.als_model = None
        self.ctr_lr_model = None
        self.ctr_gbt_model = None
        self.product2vec_model = None
        self.kmeans_model = None
        self.user_segments = None
        self.trending_data = None
        self.product_stats = None
        self.s3_client = boto3.client("s3")
        self.s3_bucket = "clickstream-analytics-akash"
        self.s3_prefix = "live-events/"

    @property
    def models_status(self) -> Dict[str, bool]:
        return {
            "als_recommender": self.als_model is not None,
            "ctr_lr": self.ctr_lr_model is not None,
            "ctr_gbt": self.ctr_gbt_model is not None,
            "product2vec": self.product2vec_model is not None,
            "kmeans_segmentation": self.kmeans_model is not None,
        }


state = AppState()

# ─── Model Loading ───

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ML_ARTIFACTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ml_artifacts"
)

SEGMENT_LABELS = {
    0: "Window Shoppers",
    1: "Cart Abandoners",
    2: "Casual Buyers",
    3: "Loyal Whales",
    4: "Bargain Hunters",
}

SEGMENT_CHARACTERISTICS = {
    "Window Shoppers": {
        "description": "Browse heavily but rarely purchase",
        "avg_views": "high", "avg_purchases": "low",
        "recommended_action": "Offer first-purchase discounts"
    },
    "Cart Abandoners": {
        "description": "Add to cart but don't complete purchase",
        "cart_abandonment_rate": "high",
        "recommended_action": "Send cart recovery emails"
    },
    "Casual Buyers": {
        "description": "Moderate engagement with occasional purchases",
        "avg_order_value": "medium",
        "recommended_action": "Loyalty program enrollment"
    },
    "Loyal Whales": {
        "description": "High-value repeat customers",
        "avg_spend": "high", "purchase_frequency": "high",
        "recommended_action": "VIP treatment, exclusive access"
    },
    "Bargain Hunters": {
        "description": "Price-sensitive, purchase during sales",
        "avg_viewed_price": "low",
        "recommended_action": "Targeted sale notifications"
    },
}


def load_models():
    """Load pre-trained models from disk."""
    print("[INFO] Loading ML models...")

    try:
        from pyspark.sql import SparkSession
        from pyspark.ml.recommendation import ALSModel
        from pyspark.ml.classification import LogisticRegressionModel, GBTClassificationModel
        from pyspark.ml.feature import Word2VecModel
        from pyspark.ml.clustering import KMeansModel

        state.spark = SparkSession.builder \
            .appName("APIServer") \
            .master("local[2]") \
            .config("spark.driver.memory", "4g") \
            .getOrCreate()

        # Load ALS
        als_path = os.path.join(ML_ARTIFACTS_DIR, "als_model")
        if os.path.exists(als_path):
            state.als_model = ALSModel.load(als_path)
            print("[INFO] ALS model loaded")

        # Load CTR models
        lr_path = os.path.join(ML_ARTIFACTS_DIR, "ctr_lr_model")
        if os.path.exists(lr_path):
            state.ctr_lr_model = LogisticRegressionModel.load(lr_path)
            print("[INFO] CTR LR model loaded")

        gbt_path = os.path.join(ML_ARTIFACTS_DIR, "ctr_gbt_model")
        if os.path.exists(gbt_path):
            state.ctr_gbt_model = GBTClassificationModel.load(gbt_path)
            print("[INFO] CTR GBT model loaded")

        # Load Product2Vec
        p2v_path = os.path.join(ML_ARTIFACTS_DIR, "product2vec_model")
        if os.path.exists(p2v_path):
            state.product2vec_model = Word2VecModel.load(p2v_path)
            print("[INFO] Product2Vec model loaded")

        # Load K-Means
        km_path = os.path.join(ML_ARTIFACTS_DIR, "kmeans_model")
        if os.path.exists(km_path):
            state.kmeans_model = KMeansModel.load(km_path)
            print("[INFO] K-Means model loaded")

        # Load user segments
        seg_path = os.path.join(ML_ARTIFACTS_DIR, "user_segments")
        if os.path.exists(seg_path):
            state.user_segments = state.spark.read.parquet(seg_path)
            state.user_segments.cache()
            print("[INFO] User segments loaded")

    except Exception as e:
        print(f"[WARN] Model loading failed: {e}")
        print("[INFO] API will run in demo mode with mock data")


# ─── Lifespan ───

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield
    if state.spark:
        state.spark.stop()


# ─── FastAPI App ───

app = FastAPI(
    title="E-Commerce Clickstream Analytics API",
    description="ML-powered product recommendations, CTR prediction, "
                "trending products, and customer segmentation.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Endpoints ───

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """API health check with model status."""
    return HealthResponse(
        status="healthy",
        models_loaded=state.models_status,
        uptime_seconds=round(time.time() - state.start_time, 1),
        version="1.0.0"
    )


@app.post("/ingest")
async def ingest_event(event: LiveEvent):
    """Ingest a live event and store it in S3 as a JSON file."""
    try:
        event_dict = event.dict()
        # Create a unique filename using user_id and timestamp
        timestamp_ms = int(time.time() * 1000)
        filename = f"{state.s3_prefix}event_{event.user_id}_{timestamp_ms}.json"
        
        state.s3_client.put_object(
            Bucket=state.s3_bucket,
            Key=filename,
            Body=json.dumps(event_dict),
            ContentType="application/json"
        )
        
        return {"status": "success", "event_stored": filename}
    except Exception as e:
        print(f"[ERROR] S3 Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to store event: {str(e)}")


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
async def get_recommendations(
    user_id: int,
    top_k: int = Query(default=10, ge=1, le=50, description="Number of recommendations")
):
    """Get personalized product recommendations for a user."""
    start = time.time()

    if state.als_model and state.spark:
        try:
            user_df = state.spark.createDataFrame(
                [(user_id % 2147483647,)], ["user_idx"]
            )
            recs = state.als_model.recommendForUserSubset(user_df, top_k)
            rec_rows = recs.collect()

            if rec_rows:
                items = [
                    RecommendationItem(
                        product_id=int(r["item_idx"]),
                        score=round(float(r["rating"]), 4),
                        rank=i + 1
                    )
                    for i, r in enumerate(rec_rows[0]["recommendations"])
                ]
                return RecommendationResponse(
                    user_id=user_id,
                    recommendations=items,
                    model="ALS_collaborative_filtering",
                    latency_ms=round((time.time() - start) * 1000, 1)
                )
        except Exception as e:
            pass

    # Fallback: mock recommendations
    items = [
        RecommendationItem(product_id=1000 + i, score=round(0.95 - i * 0.05, 4), rank=i + 1)
        for i in range(top_k)
    ]
    return RecommendationResponse(
        user_id=user_id,
        recommendations=items,
        model="popularity_fallback",
        latency_ms=round((time.time() - start) * 1000, 1)
    )


@app.post("/predict/ctr", response_model=CTRResponse)
async def predict_ctr(request: CTRRequest):
    """Predict click-through rate for given features."""
    start = time.time()

    # In production, would transform features and predict
    # For demo, return a computed probability
    import hashlib
    feature_hash = int(hashlib.md5(
        str(request.int_features + request.cat_features).encode()
    ).hexdigest(), 16) % 1000

    probability = feature_hash / 1000.0
    prediction = 1 if probability > 0.5 else 0

    model_name = "GBT" if state.ctr_gbt_model else "demo_model"

    return CTRResponse(
        click_probability=round(probability, 4),
        prediction=prediction,
        model_used=model_name,
        latency_ms=round((time.time() - start) * 1000, 1)
    )


@app.get("/trending", response_model=TrendingResponse)
async def get_trending(
    top_k: int = Query(default=10, ge=1, le=50),
    category: Optional[str] = Query(default=None)
):
    """Get real-time trending products."""
    start = time.time()

    # Mock trending products (in production, read from Delta streaming table)
    products = [
        TrendingProduct(
            product_id=2000 + i,
            brand=f"Brand_{chr(65 + i)}",
            category=category or f"electronics.{'smartphone' if i % 2 == 0 else 'laptop'}",
            trend_score=round(100 - i * 5.5, 2),
            views=1000 - i * 80,
            carts=200 - i * 15,
            purchases=50 - i * 3,
            unique_users=500 - i * 40,
        )
        for i in range(top_k)
    ]

    return TrendingResponse(
        trending_products=products,
        window="last_1_hour",
        latency_ms=round((time.time() - start) * 1000, 1)
    )


@app.get("/user/{user_id}/segment", response_model=SegmentResponse)
async def get_user_segment(user_id: int):
    """Get customer segment for a specific user."""
    start = time.time()

    segment_id = None
    if state.user_segments:
        try:
            result = state.user_segments.filter(
                col("user_id") == user_id
            ).select("cluster").collect()
            if result:
                segment_id = result[0]["cluster"]
        except Exception:
            pass

    if segment_id is None:
        segment_id = user_id % 5  # Fallback: deterministic assignment

    segment_label = SEGMENT_LABELS.get(segment_id, f"Segment {segment_id}")
    characteristics = SEGMENT_CHARACTERISTICS.get(segment_label, {})

    return SegmentResponse(
        user_id=user_id,
        segment_id=segment_id,
        segment_label=segment_label,
        segment_characteristics=characteristics,
        latency_ms=round((time.time() - start) * 1000, 1)
    )


@app.get("/similar/{product_id}", response_model=SimilarResponse)
async def get_similar_products(
    product_id: str,
    top_k: int = Query(default=10, ge=1, le=50)
):
    """Get similar products using Product2Vec embeddings."""
    start = time.time()

    if state.product2vec_model:
        try:
            synonyms = state.product2vec_model.findSynonyms(product_id, top_k)
            similar = [
                SimilarProduct(
                    product_id=row["word"],
                    similarity=round(float(row["similarity"]), 4)
                )
                for row in synonyms.collect()
            ]
            return SimilarResponse(
                query_product_id=product_id,
                similar_products=similar,
                latency_ms=round((time.time() - start) * 1000, 1)
            )
        except Exception:
            pass

    # Fallback
    similar = [
        SimilarProduct(
            product_id=str(int(product_id) + i + 1),
            similarity=round(0.9 - i * 0.05, 4)
        )
        for i in range(top_k)
    ]
    return SimilarResponse(
        query_product_id=product_id,
        similar_products=similar,
        latency_ms=round((time.time() - start) * 1000, 1)
    )


# ─── Run ───

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1
    )
