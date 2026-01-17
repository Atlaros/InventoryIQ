# src/api/main.py (Versión Corregida v2.0)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import xgboost as xgb
import pandas as pd
import pathlib
import json

# === CONFIGURACIÓN ===
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / 'models' / 'production_model.json'
DATA_PATH = BASE_DIR / 'data' / 'processed' / 'train_features.parquet'

# Variables Globales
model = None
feature_store = None
EXPECTED_FEATURES = None  # Se carga dinámicamente desde metrics.json

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, feature_store, EXPECTED_FEATURES
    print("🚀 INICIANDO API INVENTORY-IQ...")
    
    # 1. Cargar Feature List desde metrics.json
    METRICS_FILE = BASE_DIR / 'reports' / 'metrics.json'
    if not METRICS_FILE.exists():
        raise FileNotFoundError(f"❌ Archivo de métricas no encontrado: {METRICS_FILE}")
    
    with open(METRICS_FILE, 'r') as f:
        metrics = json.load(f)
        EXPECTED_FEATURES = metrics.get('features')
        if not EXPECTED_FEATURES:
            raise ValueError("❌ El archivo metrics.json no contiene la clave 'features'")
    
    print(f"✅ Features cargadas desde metrics.json: {len(EXPECTED_FEATURES)} features")
    
    # 2. Cargar Modelo
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"❌ Modelo no encontrado: {MODEL_PATH}")
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    print("✅ Modelo XGBoost cargado.")
    
    # 3. Cargar Feature Store
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"❌ Datos no encontrados: {DATA_PATH}")
    
    print("📦 Cargando e Indexando Feature Store...")
    full_df = pd.read_parquet(DATA_PATH)
    
    # Crea columna auxiliar de fecha string
    full_df['date_str'] = full_df['date'].dt.strftime('%Y-%m-%d')
    
    # --- FIX CRÍTICO AQUÍ ---
    # drop=False mantiene 'store' e 'item' como columnas, aunque sean índice.
    feature_store = full_df.set_index(['date_str', 'store', 'item'], drop=False)
    
    print(f"✅ Feature Store listo: {len(feature_store)} registros.")
    yield
    print("🛑 Apagando API...")
    del model
    del feature_store

app = FastAPI(title="InventoryIQ API", version="1.0.0", lifespan=lifespan)

class PredictionRequest(BaseModel):
    date: str
    store: int
    item: int

class PredictionResponse(BaseModel):
    date: str
    store: int
    item: int
    prediction: float

@app.get("/")
def health_check():
    return {"status": "ok", "message": "InventoryIQ Brain is Running 🧠"}

@app.get("/model/info")
def model_info():
    """Endpoint de introspección para debugging"""
    return {
        "features": EXPECTED_FEATURES,
        "model_path": str(MODEL_PATH),
        "feature_store_size": len(feature_store),
        "date_range": {
            "min": feature_store.index.get_level_values('date_str').min(),
            "max": feature_store.index.get_level_values('date_str').max()
        }
    }

@app.get("/history")
def get_history(store: int, item: int, limit: int = 30):
    """Retorna el historial de ventas para un store/item específico"""
    try:
        # Filtrar feature store por store e item
        mask = (feature_store['store'] == store) & (feature_store['item'] == item)
        history_df = feature_store[mask][['date', 'sales']].copy()
        
        # Ordenar por fecha y limitar
        history_df = history_df.sort_values('date', ascending=False).head(limit)
        
        # Convertir a formato JSON-friendly
        history_df['date'] = history_df['date'].dt.strftime('%Y-%m-%d')
        
        return history_df.to_dict(orient='records')
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo historial: {e}")


@app.post("/predict", response_model=PredictionResponse)
def predict_demand(request: PredictionRequest):
    try:
        # 1. Buscar en el Store
        key = (request.date, request.store, request.item)
        
        if key not in feature_store.index:
            raise HTTPException(status_code=404, detail=f"No tengo datos históricos (lags) para {key}")
            
        row = feature_store.loc[key]
        
        # Manejo de duplicados
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
            
        # 2. Preparar Features
        features_dict = row[EXPECTED_FEATURES].to_dict()
        features_df = pd.DataFrame([features_dict])
        
        # 3. Crear DMatrix
        dmatrix = xgb.DMatrix(features_df)
        
        # 4. Predecir
        prediction = model.predict(dmatrix)[0]
        
        return {
            "date": request.date,
            "store": request.store,
            "item": request.item,
            "prediction": round(float(prediction), 2)
        }

    except HTTPException as e:
        # ✅ EL FIX ESTÁ AQUÍ: Si ya es un error HTTP (como 404), relánzalo tal cual.
        raise e
        
    except Exception as e:
        # Si es un error desconocido (pandas fallando, etc), lanza 500.
        print(f"❌ ERROR EN SERVIDOR: {e}")
        raise HTTPException(status_code=500, detail=str(e))