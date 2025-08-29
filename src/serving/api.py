"""
Serving API:
- /recommend/{user_id} -> returns top N recommendations from collab + ranker
- logs predictions into Postgres
"""
from fastapi import FastAPI, HTTPException
import uvicorn
import joblib
import os
import json
import psycopg2
from pydantic import BaseModel
from typing import List
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from src.models.collab_matrix import SimpleMF
from src.models.ranker import predict_ranker

APP = FastAPI(title="NextPick Recommender")

# Prometheus metrics
REQ_COUNTER = Counter("api_requests_total", "Total API requests", ["endpoint"])
LATENCY = Histogram("api_latency_seconds", "Request latency", ["endpoint"])

# load models (point to models/current)
MODEL_DIR = os.getenv("MODEL_DIR", "models/current")
COLLAB_PATH = os.path.join(MODEL_DIR, "collab.pkl")
RANKER_PATH = os.path.join(MODEL_DIR, "ranker", "ranker.joblib")

collab = None
ranker = None
if os.path.exists(COLLAB_PATH):
    collab = SimpleMF.load(COLLAB_PATH)
if os.path.exists(RANKER_PATH):
    ranker = joblib.load(RANKER_PATH)

class RecommendRequest(BaseModel):
    user_id: str
    top_k: int = 10

@APP.get("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

@APP.post("/recommend")
def recommend(req: RecommendRequest):
    REQ_COUNTER.labels(endpoint="/recommend").inc()
    with LATENCY.labels(endpoint="/recommend").time():
        user_id = req.user_id
        if collab is None:
            raise HTTPException(status_code=500, detail="Collab model not loaded")
        # candidate generation
        candidates = collab.recommend(user_id, top_k=100)
        cand_df = []
        for item_id, score in candidates:
            cand_df.append({"user_id": user_id, "item_id": item_id, "collab_score": score})
        import pandas as pd
        cand_df = pd.DataFrame(cand_df)
        # if ranker exists, use it
        if ranker is not None:
            ranked = predict_ranker(ranker, cand_df)
            top = ranked.head(req.top_k)[["item_id", "rank_score"]].to_dict(orient="records")
        else:
            top = cand_df.sort_values("collab_score", ascending=False).head(req.top_k)[["item_id","collab_score"]].to_dict(orient="records")
        # optionally log to DB
        # DB details via env
        try:
            db_url = os.getenv("DATABASE_URL")
            if db_url:
                import sqlalchemy as sa
                engine = sa.create_engine(db_url)
                with engine.begin() as conn:
                    for r in top:
                        conn.execute(sa.text("INSERT INTO predictions(user_id, item_id, score, model_version) VALUES (:u,:i,:s,:m)"),
                                     {"u": user_id, "i": r.get("item_id"), "s": r.get("rank_score", r.get("collab_score")), "m": "v1"})
        except Exception as e:
            # non-fatal
            print("DB log failed:", e)
        return {"user_id": user_id, "predictions": top}

if __name__ == "__main__":
    uvicorn.run(APP, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
