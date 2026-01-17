import xgboost as xgb
import pandas as pd
import pathlib
import json

# Rutas
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / 'models' / 'production_model.json'
DATA_FILE = BASE_DIR / 'data' / 'processed' / 'train_features.parquet'

def demo_prediction():
    print("🔮 INICIANDO SISTEMA DE INFERENCIA (DEMO)")
    
    # 1. Cargar el Cerebro (El JSON)
    # Nota: Como guardamos con get_booster(), cargamos con Booster
    model = xgb.Booster()
    try:
        model.load_model(MODEL_PATH)
        print("✅ Modelo cargado correctamente en memoria.")
    except Exception as e:
        print(f"❌ Error fatal cargando el modelo: {e}")
        return

    # 2. Obtener un dato de prueba (Simulamos una petición)
    # Cargamos el dataset solo para robar una fila de ejemplo
    df = pd.read_parquet(DATA_FILE)
    
    # Tomemos una fila del futuro (2017)
    sample = df[df['year'] == 2017].sample(1)
    
    real_sales = sample['sales'].values[0]
    date = sample['date'].dt.date.values[0]
    store = sample['store'].values[0]
    item = sample['item'].values[0]
    
    # Preparamos las features (quitamos columnas que no usa el modelo)
    drop_cols = ['id', 'date', 'sales', 'year']
    features = sample.drop(columns=drop_cols)
    
    # XGBoost nativo (Booster) requiere DMatrix
    dmatrix = xgb.DMatrix(features)
    
    # 3. Predecir
    prediction = model.predict(dmatrix)[0]
    
    print("\n==========================================")
    print(f"📅 FECHA: {date} | 🏪 TIENDA: {store} | 📦 ITEM: {item}")
    print("==========================================")
    print(f"💰 Venta Real:      {real_sales}")
    print(f"🤖 Predicción IA:   {prediction:.2f}")
    print(f"📉 Diferencia:      {abs(real_sales - prediction):.2f}")
    print("==========================================")
    
    if abs(real_sales - prediction) < 15:
        print("✅ PREDICCIÓN ACEPTABLE")
    else:
        print("⚠️ ALERTA DE DESVIACIÓN (Normal en algunos casos)")

if __name__ == '__main__':
    demo_prediction()