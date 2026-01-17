import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import pathlib
import warnings

# Silenciar advertencias de Prophet (son molestas)
warnings.filterwarnings('ignore')

# === CONFIGURACIÓN ===
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
DATA_FILE = BASE_DIR / 'data' / 'processed' / 'train_features.parquet'
REPORT_DIR = BASE_DIR / 'reports' / 'figures'
REPORT_DIR.mkdir(parents=True, exist_ok=True) # Crear carpeta si no existe

def run_duel():
    print("⚔️ INICIANDO DUELO: XGBoost vs Prophet (Store 1 - Item 1) ⚔️")
    
    # 1. Cargar Datos
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"❌ No encuentro: {DATA_FILE}")
    
    df = pd.read_parquet(DATA_FILE)
    
    # 2. Filtrar "Conejillo de Indias" (Solo Tienda 1, Item 1)
    # Prophet explotaría si le damos 500 tiendas a la vez sin configurar.
    mask = (df['store'] == 1) & (df['item'] == 1)
    df_single = df[mask].copy()
    print(f"🔍 Analizando serie temporal única: {len(df_single)} días.")

    # 3. Time Split (Entrenar hasta 2016, Testear 2017)
    train = df_single[df_single['year'] < 2017].copy()
    test = df_single[df_single['year'] >= 2017].copy()
    
    print(f"📊 Train: {len(train)} días | Test: {len(test)} días")

    # ==========================================
    # 🤖 MODELO 1: XGBOOST (El Ingeniero)
    # ==========================================
    print("\n🚀 Entrenando XGBoost...")
    
    # Definir Features (Quitamos lo que no sirve para predecir)
    drop_cols = ['id', 'date', 'sales', 'year'] # Year puede confundir si no hay tendencia clara
    features = [c for c in train.columns if c not in drop_cols]
    target = 'sales'
    
    X_train = train[features]
    y_train = train[target]
    X_test = test[features]
    y_test = test[target]
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        early_stopping_rounds=50,
        random_state=42
    )
    
    # XGBoost necesita set de validación para parar si no mejora
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )
    
    # Predicción
    xgb_preds = xgb_model.predict(X_test)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
    print(f"✅ XGBoost RMSE: {xgb_rmse:.4f}")

    # ==========================================
    # 🔮 MODELO 2: PROPHET (El Adivino)
    # ==========================================
    print("\n🔮 Entrenando Prophet...")
    
    # Prophet exige columnas exactas: 'ds' (fecha) y 'y' (valor)
    prophet_train = train[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
    prophet_test = test[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
    
    # Instanciar y añadir festivos de EE.UU incorporados
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m.add_country_holidays(country_name='US')
    
    m.fit(prophet_train)
    
    # Predecir el futuro (el periodo de test)
    future = prophet_test[['ds']]
    forecast = m.predict(future)
    
    prophet_preds = forecast['yhat'].values
    prophet_rmse = np.sqrt(mean_squared_error(prophet_test['y'], prophet_preds))
    print(f"✅ Prophet RMSE: {prophet_rmse:.4f}")

    # ==========================================
    # 📈 VISUALIZACIÓN DEL RESULTADO
    # ==========================================
    plt.figure(figsize=(15, 7))
    
    # Graficar solo 2017 (Real vs Predicciones)
    dates = test['date']
    plt.plot(dates, y_test, label='Ventas Reales', color='black', alpha=0.6)
    plt.plot(dates, xgb_preds, label=f'XGBoost (RMSE={xgb_rmse:.2f})', color='blue', linestyle='--')
    plt.plot(dates, prophet_preds, label=f'Prophet (RMSE={prophet_rmse:.2f})', color='red', linestyle='-.')
    
    plt.title('DUELO FINAL: Store 1 - Item 1 (Predicción Año 2017)', fontsize=16)
    plt.xlabel('Fecha')
    plt.ylabel('Ventas')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Guardar
    output_path = REPORT_DIR / 'duel_result.png'
    plt.savefig(output_path)
    print(f"\n🖼️ Gráfico guardado en: {output_path}")
    
    # Ganador
    if xgb_rmse < prophet_rmse:
        print("\n🏆 GANADOR: XGBoost")
    else:
        print("\n🏆 GANADOR: Prophet")

if __name__ == '__main__':
    run_duel()