# src/models/train_model.py
"""
XGBoost Training Pipeline - InventoryIQ
Entrena modelo de predicción de ventas con validación temporal.
"""

import pandas as pd
import numpy as np
import pathlib
import pickle
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# === CONFIGURACIÓN DE RUTAS ===
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'data' / 'processed'
MODEL_DIR = BASE_DIR / 'models'
INPUT_FILE = DATA_DIR / 'train_features.parquet'
MODEL_FILE = MODEL_DIR / 'xgboost_sales_model.pkl'

def load_data() -> pd.DataFrame:
    """Carga el dataset de features procesado."""
    print(f"📥 Cargando datos desde: {INPUT_FILE}")
    
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"❌ No se encuentra el archivo: {INPUT_FILE}\n"
            f"   Ejecuta primero: python -m src.features.build_features"
        )
    
    df = pd.read_parquet(INPUT_FILE)
    print(f"✅ Datos cargados: {df.shape}")
    print(f"   Columnas: {list(df.columns)}")
    
    return df

def time_series_split(df: pd.DataFrame, split_year: int = 2017):
    """
    Divide los datos cronológicamente para validación temporal.
    
    Args:
        df: DataFrame con columna 'year'
        split_year: Año de corte (< split_year = train, >= split_year = test)
    
    Returns:
        train_df, test_df
    """
    print(f"\n🔪 División Temporal (Time Series Split):")
    print(f"   Train: año < {split_year}")
    print(f"   Test:  año >= {split_year}")
    
    train_df = df[df['year'] < split_year].copy()
    test_df = df[df['year'] >= split_year].copy()
    
    print(f"   ✅ Train set: {train_df.shape[0]} registros")
    print(f"   ✅ Test set:  {test_df.shape[0]} registros")
    
    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError(
            f"❌ Split inválido. Train: {len(train_df)}, Test: {len(test_df)}\n"
            f"   Verifica que el año {split_year} exista en tus datos."
        )
    
    return train_df, test_df

def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Separa features (X) y target (y) para train y test.
    
    Excluye: 'sales' (target), 'date' (temporal), 'id' (no predictivo)
    """
    print("\n🎯 Preparando Features y Target:")
    
    # Columnas a excluir
    exclude_cols = ['sales', 'date', 'id']
    
    # Obtener columnas de features (todas menos las excluidas)
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    print(f"   Target: 'sales'")
    print(f"   Features ({len(feature_cols)}): {feature_cols}")
    
    # Separar X e y
    X_train = train_df[feature_cols]
    y_train = train_df['sales']
    
    X_test = test_df[feature_cols]
    y_test = test_df['sales']
    
    print(f"   ✅ X_train: {X_train.shape}")
    print(f"   ✅ y_train: {y_train.shape}")
    print(f"   ✅ X_test:  {X_test.shape}")
    print(f"   ✅ y_test:  {y_test.shape}")
    
    return X_train, y_train, X_test, y_test, feature_cols

def train_xgboost(X_train, y_train):
    """
    Entrena modelo XGBoost con hiperparámetros optimizados.
    """
    print("\n🚀 Entrenando XGBoost Regressor...")
    
    model = XGBRegressor(
        n_estimators=100,      # Número de árboles
        learning_rate=0.1,     # Tasa de aprendizaje
        max_depth=5,           # Profundidad máxima de árboles
        random_state=42,       # Reproducibilidad
        n_jobs=-1              # Usar todos los cores
    )
    
    print(f"   Parámetros: n_estimators=100, learning_rate=0.1, max_depth=5")
    
    model.fit(X_train, y_train)
    
    print(f"✅ Modelo entrenado exitosamente!")
    
    return model

def evaluate_model(model, X_test, y_test, test_df):
    """
    Evalúa el modelo y muestra métricas + comparación.
    """
    print("\n📊 Evaluación del Modelo:")
    
    # Predicciones
    y_pred = model.predict(X_test)
    
    # Calcular RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\n{'='*60}")
    print(f"   RMSE del Modelo: {rmse:.2f}")
    print(f"{'='*60}")
    
    # Crear DataFrame de comparación
    comparison = pd.DataFrame({
        'Fecha': test_df['date'].values,
        'Venta_Real': y_test.values,
        'Venta_Predicha': y_pred,
        'Diferencia': y_test.values - y_pred
    })
    
    print("\n🔍 Comparación Real vs Predicho (primeras 5 filas):")
    print("-" * 80)
    print(comparison.head(5).to_string(index=False))
    
    print("\n📈 Estadísticas de Error:")
    print(f"   Error Promedio: {comparison['Diferencia'].mean():.2f}")
    print(f"   Error Absoluto Promedio: {comparison['Diferencia'].abs().mean():.2f}")
    print(f"   Error Máximo: {comparison['Diferencia'].abs().max():.2f}")
    
    return rmse, comparison

def save_model(model, feature_cols):
    """
    Guarda el modelo entrenado y metadatos.
    """
    print(f"\n💾 Guardando modelo en: {MODEL_FILE}")
    
    # Crear directorio si no existe
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Guardar modelo y metadatos
    model_data = {
        'model': model,
        'feature_columns': feature_cols,
        'model_type': 'XGBRegressor',
        'params': model.get_params()
    }
    
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"✅ Modelo guardado exitosamente!")

if __name__ == '__main__':
    print("=" * 80)
    print("🤖 XGBOOST TRAINING PIPELINE - InventoryIQ")
    print("=" * 80)
    
    try:
        # 1. Cargar datos
        df = load_data()
        
        # 2. Split temporal
        train_df, test_df = time_series_split(df, split_year=2017)
        
        # 3. Preparar features
        X_train, y_train, X_test, y_test, feature_cols = prepare_features(train_df, test_df)
        
        # 4. Entrenar modelo
        model = train_xgboost(X_train, y_train)
        
        # 5. Evaluar modelo
        rmse, comparison = evaluate_model(model, X_test, y_test, test_df)
        
        # 6. Guardar modelo
        save_model(model, feature_cols)
        
        print("\n" + "=" * 80)
        print("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
        print("=" * 80)
        print(f"\n📌 Resumen:")
        print(f"   - RMSE: {rmse:.2f}")
        print(f"   - Modelo guardado en: {MODEL_FILE}")
        print(f"   - Features utilizadas: {len(feature_cols)}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise
