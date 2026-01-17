## src/features/build_features.py (Versión Corregida v2.0)
import pandas as pd
import holidays
import pathlib
from src.data.loader import fetch_data

# === CONFIGURACIÓN DE RUTAS ===
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
OUTPUT_FILE = PROCESSED_DIR / 'train_features.parquet'

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Genera partes de la fecha (Año, Mes, Día, etc.)"""
    if 'date' not in df.columns:
        raise ValueError("❌ Falta la columna 'date'")
    
    df = df.copy()
    # Vectorización pura (Cero bucles)
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    print(f"✅ Fechas desglosadas (Year, Month, etc).")
    return df

def add_holidays(df: pd.DataFrame) -> pd.DataFrame:
    """Detecta festivos en USA corrigiendo el formato de fecha."""
    print("🎉 Detectando festivos (US)...")
    
    # 1. Obtener los años únicos del dataset para optimizar la librería
    years = df['date'].dt.year.unique()
    
    # 2. Generar diccionario de festivos explícito
    us_holidays = holidays.US(years=years)
    
    # 3. Comparación vectorizada (Date vs Date)
    # Convertimos la columna timestamp a date object para comparar
    dates_series = df['date'].dt.date 
    
    df['is_holiday'] = dates_series.isin(us_holidays).astype(int)
    
    count = df['is_holiday'].sum()
    if count == 0:
        print("⚠️ ADVERTENCIA: No se detectaron festivos. Revisa la librería holidays.")
    else:
        print(f"✅ Festivos encontrados: {count} días marcados.")
    
    return df

def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    print("⏱️ Creando Lags y Rolling Means (esto tarda un poco)...")
    
    # Ordenar es vital para shift
    df = df.sort_values(['store', 'item', 'date'])
    
    # Groupby para no mezclar tiendas
    # Lag 1: Venta de ayer
    df['lag_1'] = df.groupby(['store', 'item'])['sales'].shift(1)
    
    # Lag 7: Venta de hace una semana (mismo día de la semana)
    df['lag_7'] = df.groupby(['store', 'item'])['sales'].shift(7)
    
    # Rolling Mean: Promedio de los últimos 7 días (excluyendo hoy)
    # Usamos shift(1) para evitar Data Leakage
    df['rmean_7'] = df.groupby(['store', 'item'])['sales'].transform(
        lambda x: x.shift(1).rolling(7).mean()
    )
    
    # Limpiar NaNs (los primeros 7 días de cada serie quedan vacíos)
    before = len(df)
    df = df.dropna()
    print(f"🧹 Limpieza: {before - len(df)} filas eliminadas por falta de historial.")
    
    return df

if __name__ == '__main__':
    print("========================================")
    print("🚀 FEATURE ENGINEERING PIPELINE")
    print("========================================")
    
    # 1. Cargar
    df = fetch_data()
    
    # 2. Fechas
    df = generate_features(df)
    
    # 3. Festivos
    df = add_holidays(df)
    
    # 4. Lags (Memoria)
    df = add_lags(df)
    
    # 5. GUARDADO (CRÍTICO)
    # Asegurar que existe el directorio
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 Guardando dataset maestro en: {OUTPUT_FILE}")
    df.to_parquet(OUTPUT_FILE, index=False)
    
    print("\n✅ REPORTE FINAL:")
    print(f"Shape: {df.shape}")
    print(f"Columnas: {list(df.columns)}")
    print("Muestra de festivos:")
    print(df[df['is_holiday'] == 1][['date', 'is_holiday']].head(3))