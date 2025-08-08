# Retail Recommendation Project

Bu proje RetailRocket veri seti kullanılarak hazırlanmış basit bir öneri sistemi pipeline'ıdır.

## Klasör Yapısı
```
retail_recommendation_app/
├─ data/
│  ├─ raw/
│  ├─ processed/
├─ models/
├─ app/
│  └─ api_server.py      # FastAPI servisi
├─ train_pipeline.py         # Pipeline (data prep + training)
├─ requirements.txt
└─ README.md
```

## Kullanım
1. `python main.py` komutu ile veriyi hazırla ve modeli eğit.
2. `uvicorn app.api_server:app --reload` komutu ile API'yi çalıştır.
3. `http://127.0.0.1:8000/docs` adresinden API'yi test et.
