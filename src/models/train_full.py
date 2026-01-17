import pandas as pd
import numpy as np
import xgboost as xgb
import pathlib
import json
from sklearn.metrics import mean_squared_error

# Configuración
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
DATA_FILE = BASE_DIR / 'data' / 'processed' / 'train_features.parquet'
MODEL_DIR = BASE_DIR / 'models'
MODEL_PATH = MODEL_DIR / 'production_model.json'
METRICS_PATH = BASE_DIR / 'reports' / 'metrics.json'

MODEL_DIR.mkdir(parents=True, exist_ok=True)

def train_production_model():
    print("🌍 INICIANDO ENTRENAMIENTO A ESCALA GLOBAL (Todas las Tiendas/Items)")
    
    # 1. Cargar Todo
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"❌ No existe el archivo: {DATA_FILE}")
        
    df = pd.read_parquet(DATA_FILE)
    print(f"📦 Dataset Cargado: {df.shape}")

    # 2. Split Estratégico (Train < 2017, Test >= 2017)
    train = df[df['year'] < 2017]
    test = df[df['year'] >= 2017]
    
    # Features (Quitamos ruido)
    drop_cols = ['id', 'date', 'sales', 'year']
    features = [c for c in train.columns if c not in drop_cols]
    target = 'sales'
    
    print(f"🎯 Features ({len(features)}): {features}")
    
    X_train = train[features]
    y_train = train[target]
    X_test = test[features]
    y_test = test[target]
    
    # 3. Configuración del Modelo (FIXED para XGBoost 2.0+)
    # early_stopping_rounds AHORA va aquí, no en fit()
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=50, # <--- MOVIDO AQUÍ
        n_jobs=-1,
        random_state=42
    )
    
    print("🔥 Entrenando... (Esto tomará 1-2 minutos)...")
    
    # En fit() solo pasamos el eval_set para que early_stopping tenga qué medir
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100
    )
    
    # 4. Evaluación Global
    print("\n📊 Evaluando...")
    preds = model.predict(X_test)
    
    # Cálculo manual de RMSE (más compatible entre versiones de sklearn)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"\n🌟 RMSE GLOBAL (Promedio de 500 series): {rmse:.4f}")
    
    # 5. Guardar Artefacto
    # Usamos save_model (formato JSON universal)
    # ✅ ESTO ES ROBUSTO (Guarda el core del modelo)
    model.get_booster().save_model(MODEL_PATH)
    print(f"✅ Modelo guardado en: {MODEL_PATH}")
    
    # Guardar métricas
    metrics = {"rmse": float(rmse), "features": features}
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f)

if __name__ == '__main__':
    train_production_model()