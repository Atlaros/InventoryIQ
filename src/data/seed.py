import os
import pathlib
import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv
from tqdm import tqdm  # Barra de progreso

# 1. Configuración de Rutas (Pathlib para evitar errores de OS)
# Estamos en src/data/seed.py, subimos 3 niveles para llegar a la raíz
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / 'data' / 'raw' / 'train.csv'
ENV_PATH = BASE_DIR / '.env'

# 2. Carga de Secretos
if not load_dotenv(ENV_PATH):
    print("⚠️ ADVERTENCIA: No se encontró el archivo .env")

URL = os.getenv("SUPABASE_URL")
KEY = os.getenv("SUPABASE_KEY")

if not URL or not KEY:
    raise ValueError("❌ Error: Faltan credenciales de Supabase en .env")

# 3. Cliente Supabase
supabase: Client = create_client(URL, KEY)

def seed_database():
    print("==========================================")
    print("🚀 INICIANDO PROTOCOLO DE CARGA (SEEDER)")
    print("==========================================")

    # Verificación de archivo
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"❌ No encuentro el CSV en: {DATA_PATH}")
    
    print(f"📂 Leyendo dataset: {DATA_PATH}...")
    
    # Lectura del CSV
    try:
        df = pd.read_csv(DATA_PATH)
        # Convertir fecha a string formato ISO para SQL
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        print(f"✅ Dataset cargado en memoria: {len(df)} filas.")
    except Exception as e:
        raise Exception(f"❌ Error leyendo CSV: {e}")

    # Configuración de lotes (Batch processing)
    BATCH_SIZE = 1000
    total_rows = len(df)
    
    print(f"📡 Conectando a Supabase ({URL})...")
    print("⏳ Subiendo datos en lotes de 1000... (Esto tomará unos minutos)")

    # Bucle de inserción con barra de progreso
    for i in tqdm(range(0, total_rows, BATCH_SIZE), desc="Progreso"):
        # Cortar el dataframe (slicing)
        batch = df.iloc[i : i + BATCH_SIZE]
        
        # Convertir a lista de diccionarios (formato JSON para API)
        records = batch.to_dict(orient='records')
        
        try:
            # Insertar en Supabase
            supabase.table('sales').insert(records).execute()
        except Exception as e:
            print(f"\n❌ Error insertando lote {i}: {e}")
            # Opcional: break para detener si hay error crítico
            
    print("\n==========================================")
    print("🏁 CARGA COMPLETADA EXITOSAMENTE")
    print("==========================================")

if __name__ == '__main__':
    seed_database()