# src/data/loader.py (Versión Resiliente v3.0)
import os
import time
import pathlib
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv
from tqdm import tqdm

# Configuración
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
CACHE_DIR = BASE_DIR / 'data' / 'processed'
ENV_PATH = BASE_DIR / '.env'

load_dotenv(ENV_PATH)

def fetch_data(table: str = "sales", force_db: bool = False) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{table}_cache.parquet"

    if cache_path.exists() and not force_db:
        print(f"⚡ Cargando desde caché local: {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"🌐 Conectando a Supabase para descargar '{table}'...")
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    
    # Validación de credenciales
    if not url or not key:
        raise EnvironmentError("❌ Faltan SUPABASE_URL o SUPABASE_KEY en .env")
    
    supabase = create_client(url, key)

    all_rows = []
    chunk_size = 1000
    offset = 0
    
    # === CONFIGURACIÓN DE RESILIENCIA ===
    MAX_RETRIES = 5  # Intentos antes de rendirse
    
    print("📥 Iniciando descarga paginada con Auto-Retry...")
    pbar = tqdm(desc="Filas descargadas")
    
    while True:
        # Bucle de Reintentos para cada bloque
        batch_data = None
        for attempt in range(MAX_RETRIES):
            try:
                # Intentamos descargar el bloque actual
                response = supabase.table(table).select("*").range(offset, offset + chunk_size - 1).execute()
                batch_data = response.data
                break # ¡Éxito! Salimos del bucle de reintentos
            except Exception as e:
                wait_time = 2 ** attempt # Espera exponencial: 1s, 2s, 4s, 8s...
                pbar.write(f"⚠️ Error en offset {offset} (Intento {attempt+1}/{MAX_RETRIES}): {e}. Reintentando en {wait_time}s...")
                time.sleep(wait_time)
        
        # Si después de 5 intentos sigue fallando, explotamos controladamente
        if batch_data is None:
            raise ConnectionError(f"❌ Imposible descargar bloque en offset {offset} tras {MAX_RETRIES} intentos.")

        if not batch_data:
            break # Fin de los datos
            
        all_rows.extend(batch_data)
        batch_len = len(batch_data)
        offset += batch_len
        pbar.update(batch_len)
        
        if batch_len < chunk_size:
            break
            
    pbar.close()
    
    if not all_rows:
        raise ValueError("❌ Base de datos vacía.")

    print(f"✅ Descarga completada: {len(all_rows)} filas.")
    df = pd.DataFrame(all_rows)

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

    print(f"💾 Guardando caché: {cache_path}")
    df.to_parquet(cache_path, index=False)
    return df

if __name__ == '__main__':
    try:
        # Force_db=True para obligarlo a probar la descarga completa de nuevo
        df = fetch_data(force_db=True)
        print(f"\n✅ REPORTE FINAL: {df.shape}")
    except Exception as e:
        print(f"\n❌ FATAL: {e}")