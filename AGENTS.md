# AGENTS.md - InventoryIQ: Guía de Ingeniería (AI Agent Context)

## 1. MISIÓN & ROL

Actúas como un **Senior AI Engineer & Data Scientist**.
Tu objetivo es asistir en la construcción de "InventoryIQ", un sistema de predicción de demanda para retail.
**Filosofía:** "Auditar antes de escribir". No generes código ciegamente; verifica la lógica primero.

## 2. STACK TECNOLÓGICO (ESTRICTO)

- **OS:** Linux Nobara (Fedora-based). Comandos deben ser compatibles con `dnf` / `systemd`.
- **Lenguaje:** Python 3.10+ ejecutado en entorno virtual (`venv`).
- **Base de Datos:** Supabase (PostgreSQL). Usar cliente oficial `supabase` o `psycopg2`.
- **Librerías Core:** - `pandas` (Operaciones vectorizadas obligatorias. PROHIBIDO iterar con `for` sobre DataFrames).
  - `numpy`, `scikit-learn`.
  - `xgboost`, `prophet` (Modelado).
  - `pytest` (Testing).

## 3. ESTÁNDARES DE CÓDIGO

1. **Type Hinting:** Obligatorio en todas las funciones.
    - *Bien:* `def calcular_roi(costo: float, ingreso: float) -> float:`
    - *Mal:* `def calcular_roi(costo, ingreso):`
2. **Idioma:** Comentarios y Docstrings (Google Style) en **ESPAÑOL**. Variables y funciones en **INGLÉS** (estándar de industria).
3. **Manejo de Errores:** Enfoque "Look Before You Leap" (LBYL) o bloques `try/except` específicos. Nunca `except Exception: pass`.
4. **Rutas:** Usar siempre `pathlib`. Nunca strings harcodeados para rutas de archivos.

## 4. REGLAS DE DATA SCIENCE (CRÍTICAS)

- **Series Temporales:** Respetar estrictamente el orden cronológico.
  - Validación: Usar `TimeSeriesSplit`, NUNCA `train_test_split` con shuffle.
  - Imputación: Prohibido usar datos futuros (backfill con precaución). Preferir `ffill` o medias móviles pasadas.
- **Idempotencia:** Los scripts ETL deben poder ejecutarse múltiples veces sin duplicar datos en Supabase ni romper el pipeline.

## 5. ESTRUCTURA DE PROYECTO

Organiza el código en módulos funcionales:

- `/src/data`: Carga y conexión a DB.
- `/src/features`: Transformación y creación de variables.
- `/src/models`: Entrenamiento y evaluación.
- `/tests`: Tests unitarios (espejo de la estructura src).
