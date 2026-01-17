# tests/test_api.py
from fastapi.testclient import TestClient
from src.api.main import app
import pytest

# --- FIX CRÍTICO: USAR FIXTURE PARA EL LIFESPAN ---
@pytest.fixture(scope="module")
def client():
    # El bloque 'with' fuerza a ejecutar el evento startup (carga de datos)
    with TestClient(app) as c:
        yield c
    # Al salir del 'with', se ejecuta el shutdown (limpieza)

def test_health_check(client):
    """Verifica que la API está viva (Status 200)"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "InventoryIQ Brain is Running 🧠"}

def test_predict_endpoint_structure(client):
    """
    Verifica que el endpoint /predict acepta el JSON correcto
    y devuelve la estructura esperada.
    """
    # Usamos una fecha que sabemos que existe (2017)
    payload = {
        "date": "2017-06-15",
        "store": 1,
        "item": 1
    }
    
    response = client.post("/predict", json=payload)
    
    # 1. Debe responder 200 OK
    assert response.status_code == 200
    
    # 2. Debe devolver un JSON
    data = response.json()
    
    # 3. Debe contener la predicción
    assert "prediction" in data
    assert isinstance(data["prediction"], float)
    
    # 4. La predicción debe ser lógica (no negativa)
    assert data["prediction"] >= 0

def test_predict_error_handling(client):
    """Verifica que la API lance error 404 si pedimos una fecha fuera de rango"""
    payload = {
        "date": "1990-01-01", # Fecha prehistórica
        "store": 1,
        "item": 1
    }
    response = client.post("/predict", json=payload)
    
    # Debe fallar controladamente (404 Not Found)
    assert response.status_code == 404
    assert "detail" in response.json()