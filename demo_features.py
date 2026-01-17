#!/usr/bin/env python
"""
Demo script para verificar build_features.py sin descargar datos de Supabase.
Usa datos sintéticos para demostrar la funcionalidad.
"""
import sys
import pandas as pd
from pathlib import Path

# Agregar el directorio raíz al PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.features.build_features import generate_features

print("=" * 60)
print("DEMO: FEATURE ENGINEERING - InventoryIQ")
print("=" * 60)

# Crear datos sintéticos de prueba
print("\n📊 Creando datos de prueba...")
sample_dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='W')
df_test = pd.DataFrame({
    'date': sample_dates,
    'product_id': range(1, len(sample_dates) + 1),
    'sales': [100 + i * 10 for i in range(len(sample_dates))]
})
print(f"   Shape original: {df_test.shape}")

# Aplicar feature engineering
print("\n🔧 Aplicando feature engineering...")
df_result = generate_features(df_test)
print(f"   Shape con features: {df_result.shape}")

# Verificación visual
print("\n🔍 Verificación con 5 muestras aleatorias:")
print("-" * 60)
sample = df_result[['date', 'year', 'month', 'day', 'day_of_week', 'is_weekend', 'quarter']].sample(5)

day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

for idx, row in sample.iterrows():
    day_name = day_names[row['day_of_week']]
    weekend_status = "✓ WEEKEND" if row['is_weekend'] == 1 else "  Weekday"
    
    print(f"Date: {row['date'].strftime('%Y-%m-%d')} ({day_name:9s}) | "
          f"Year: {row['year']} | Month: {row['month']:2d} | Day: {row['day']:2d} | "
          f"DOW: {row['day_of_week']} | Q{row['quarter']} | {weekend_status}")

# Validar que los fines de semana están correctamente marcados
print("\n✅ Validación de is_weekend:")
weekends = df_result[df_result['is_weekend'] == 1]
print(f"   Total registros: {len(df_result)}")
print(f"   Fines de semana detectados: {len(weekends)}")
print(f"   Días de semana únicos en weekends: {sorted(weekends['day_of_week'].unique())}")

if set(weekends['day_of_week'].unique()) == {5, 6}:
    print("   ✓ Validación exitosa: Solo sábados (5) y domingos (6)")
else:
    print("   ✗ Error en la detección de fines de semana")

print("\n" + "=" * 60)
print("✅ Demo completada exitosamente!")
print("=" * 60)
print("\n💡 Para usar con datos reales de Supabase:")
print("   source .venv/bin/activate")
print("   python -m src.features.build_features")
