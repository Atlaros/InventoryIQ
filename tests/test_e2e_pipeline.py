# tests/test_e2e_pipeline.py
"""
Test de integración End-to-End que verifica el pipeline completo:
ETL -> Feature Engineering -> Model Training -> API Inference
"""
import pytest
import pandas as pd
import pathlib
import json
from src.data.loader import fetch_data
from src.features.build_features import generate_features, add_holidays, add_lags
from fastapi.testclient import TestClient
from src.api.main import app

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
MODEL_PATH = BASE_DIR / 'models' / 'production_model.json'
METRICS_PATH = BASE_DIR / 'reports' / 'metrics.json'

@pytest.fixture(scope="module")
def api_client():
    """Cliente de API con lifespan completo"""
    with TestClient(app) as client:
        yield client

def test_e2e_data_loading():
    """Test 1: Verificar que los datos se pueden cargar desde cache o Supabase"""
    df = fetch_data()
    
    # Verificaciones básicas
    assert df is not None, "El DataFrame no debe ser None"
    assert len(df) > 0, "El DataFrame debe tener filas"
    assert 'date' in df.columns, "Debe existir columna 'date'"
    assert 'sales' in df.columns, "Debe existir columna 'sales'"
    
    print(f"✅ Test E2E 1/5: Datos cargados correctamente ({len(df)} filas)")

def test_e2e_feature_engineering():
    """Test 2: Verificar que el feature engineering genera las columnas esperadas"""
    df = fetch_data()
    
    # Aplicar features
    df = generate_features(df)
    df = add_holidays(df)
    
    # Verificar que las columnas existen
    expected_date_features = ['year', 'month', 'day', 'day_of_week', 'quarter', 'is_weekend']
    for feat in expected_date_features:
        assert feat in df.columns, f"Falta la feature: {feat}"
    
    assert 'is_holiday' in df.columns, "Falta la feature de festivos"
    
    print(f"✅ Test E2E 2/5: Feature engineering aplicado correctamente")

def test_e2e_model_exists():
    """Test 3: Verificar que el modelo entrenado existe y es válido"""
    assert MODEL_PATH.exists(), f"El modelo no existe en {MODEL_PATH}"
    
    # Verificar que metrics.json también existe
    assert METRICS_PATH.exists(), f"El archivo metrics.json no existe en {METRICS_PATH}"
    
    with open(METRICS_PATH, 'r') as f:
        metrics = json.load(f)
    
    assert 'features' in metrics, "metrics.json debe contener 'features'"
    assert 'rmse' in metrics, "metrics.json debe contener 'rmse'"
    assert len(metrics['features']) > 0, "La lista de features no debe estar vacía"
    
    print(f"✅ Test E2E 3/5: Modelo y métricas existen (RMSE: {metrics['rmse']:.2f})")

def test_e2e_feature_store_exists():
    """Test 4: Verificar que el feature store (train_features.parquet) existe"""
    feature_store_path = PROCESSED_DIR / 'train_features.parquet'
    assert feature_store_path.exists(), f"Feature store no existe en {feature_store_path}"
    
    df = pd.read_parquet(feature_store_path)
    assert len(df) > 0, "El feature store debe tener datos"
    
    # Verificar que tiene las features de lag
    assert 'lag_1' in df.columns, "Falta lag_1"
    assert 'lag_7' in df.columns, "Falta lag_7"
    assert 'rmean_7' in df.columns, "Falta rmean_7"
    
    print(f"✅ Test E2E 4/5: Feature store válido ({len(df)} filas)")

def test_e2e_api_prediction_pipeline(api_client):
    """Test 5: Verificar que la API puede hacer predicciones usando todos los componentes"""
    # Este test verifica el flujo completo:
    # 1. API carga modelo desde disco
    # 2. API carga feature store
    # 3. API carga lista de features desde metrics.json
    # 4. API hace predicción
    
    payload = {
        "date": "2017-06-15",
        "store": 1,
        "item": 1
    }
    
    response = api_client.post("/predict", json=payload)
    
    # Verificar respuesta exitosa
    assert response.status_code == 200, f"API falló: {response.text}"
    
    data = response.json()
    assert "prediction" in data, "La respuesta debe contener 'prediction'"
    assert data["prediction"] >= 0, "La predicción debe ser no negativa"
    
    print(f"✅ Test E2E 5/5: Predicción exitosa ({data['prediction']} unidades)")

if __name__ == "__main__":
    """Permite ejecutar los tests directamente"""
    print("\n" + "="*60)
    print("🧪 EJECUTANDO TESTS E2E DEL PIPELINE COMPLETO")
    print("="*60 + "\n")
    
    pytest.main([__file__, "-v", "--tb=short"])
