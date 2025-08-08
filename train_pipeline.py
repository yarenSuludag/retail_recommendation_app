"""
train_pipeline.py
----------------------------------------
Bu dosyada öneri sisteminin eğitim pipeline'ını yazdım.
Amaç: events.csv'deki etkileşimlerden anlamlı kullanıcı-ürün ilişkileri çıkarıp, satın alma tahmini yapan bir model eğitmek.
"""

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

# Ham veri, işlenmiş veri ve modelin kaydedileceği yollar
RAW_EVENTS = "data/raw/events.csv"
PROCESSED = "data/processed/features.csv"
MODEL_PATH = "models/model.pkl"

def prepare_data():
    """
    Burada events.csv'den kullanıcı-ürün bazlı özet istatistikler çıkarıyorum.
    Özellikleri sade ama anlamlı tuttum: görüntüleme, sepete ekleme sayısı, sepette kalma süresi gibi.
    Ayrıca ürünün genel popülerliğini de bir feature olarak ekledim.
    """

    print(">>> events.csv yükleniyor...")
    events = pd.read_csv(RAW_EVENTS)

    # event türlerini netleştirmek için label ekliyorum
    events['label'] = (events['event'] == 'transaction').astype(int)

    # Ürün genel popülerliği: kaç farklı kişi görüntülemiş (global feature)
    item_popularity = events[events['event'] == 'view'].groupby('itemid')['visitorid'].nunique().reset_index()
    item_popularity.columns = ['itemid', 'unique_views']

    print(">>> Kullanıcı-ürün bazlı etkileşimler toplanıyor...")
    feats = events.groupby(['visitorid', 'itemid']).agg(
        view_count=('event', lambda x: (x == 'view').sum()),
        add_count=('event', lambda x: (x == 'addtocart').sum()),
        buy_count=('event', lambda x: (x == 'transaction').sum()),
        last_ts=('timestamp', 'max')
    ).reset_index()

    # sepette kalma süresi = timestamp'i gün cinsine çevirdim
    feats['days_in_cart'] = feats['last_ts'] / (1000 * 60 * 60 * 24)

    # label: satın alındıysa 1
    feats['label'] = (feats['buy_count'] > 0).astype(int)

    # Ürünün global popülerliğini ekliyorum
    feats = feats.merge(item_popularity, on='itemid', how='left')
    feats['unique_views'] = feats['unique_views'].fillna(0)

    # Not: bazı sütunları artık kullanmıyorum
    feats = feats[['visitorid', 'itemid', 'view_count', 'add_count', 'days_in_cart', 'unique_views', 'label']]

    # class imbalance azaltmak için 1'e 3 oranında undersampling yaptım
    df_pos = feats[feats['label'] == 1]
    df_neg = feats[feats['label'] == 0].sample(n=min(len(df_pos)*3, len(feats[feats['label'] == 0])), random_state=42)
    df_balanced = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Kayıt etmek için klasörü oluşturuyorum
    os.makedirs("data/processed", exist_ok=True)
    df_balanced.to_csv(PROCESSED, index=False)
    print(f">>> Feature set kaydedildi: {PROCESSED}")

    return df_balanced

def train_model():
    """
    Veriyi alıp hem RandomForest hem de LogisticRegression ile modeli eğitiyorum.
    Hedefim: F1 skoruna göre en iyi modeli bulup onu kaydetmek.
    """

    print(">>> Feature set yükleniyor...")
    df = pd.read_csv(PROCESSED)

    X = df[['view_count', 'add_count', 'days_in_cart', 'unique_views']]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # class_weight='balanced' özellikle LogisticRegression için etkili oluyor
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced')
    }

    best_model = None
    best_f1 = 0.0

    print(">>> Modeller eğitiliyor...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"{name} | Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

        if f1 > best_f1:
            best_model = model
            best_f1 = f1

    # En iyi modeli kaydediyorum
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f">>> En iyi model kaydedildi: {MODEL_PATH}")

if __name__ == "__main__":
    prepare_data()
    train_model()
