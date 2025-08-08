from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(
    title="Retail Recommendation API",
    version="0.1.0",
    description="Kullanıcı sepetine göre ürün öneren bir sistem. Modelle tahmin yapıyor."
)

# CORS ayarı (frontend erişimi için)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Statik dosyalar (HTML, CSS, JS) için mount işlemi
app.mount("/static", StaticFiles(directory="static"), name="static")

# Ana sayfaya gelen isteklerde index.html döner
@app.get("/")
def root():
    return FileResponse("static/index.html")

# Model dosyası
MODEL_PATH = "models/model.pkl"
model = None

# Veri modelleri
class CartItem(BaseModel):
    itemid: int
    view_count: int
    add_count: int
    days_in_cart: float
    unique_views: int

class RecommendRequest(BaseModel):
    user_id: int
    cart: List[CartItem]
    top_k: int = 5

# Modeli sunucu başlarken yükle
@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f">>> Model başarıyla yüklendi: {MODEL_PATH}")
    else:
        print(">>> Model bulunamadı. Lütfen önce train_pipeline.py'yi çalıştır.")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/recommend")
def recommend(req: RecommendRequest):
    if model is None:
        return {"error": "Model yüklenmedi."}

    cart_df = pd.DataFrame([item.dict() for item in req.cart])
    feature_cols = ["view_count", "add_count", "days_in_cart", "unique_views"]
    X = cart_df[feature_cols]

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    item_scores = list(zip(cart_df["itemid"].tolist(), probabilities.tolist()))
    item_scores.sort(key=lambda x: x[1], reverse=True)
    top_items = [item_id for item_id, _ in item_scores[:req.top_k]]

    return {
        "user_id": req.user_id,
        "recommended_items": top_items,
        "total_candidates": len(req.cart),
        "recommended_count": len(top_items),
        "item_scores": item_scores
    }
