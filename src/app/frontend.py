# src/app/frontend.py
import streamlit as st
import requests
import datetime
import plotly.graph_objects as go
import pandas as pd
import os

# ===========================
# CONFIGURACIÓN DE LA PÁGINA
# ===========================
st.set_page_config(
    page_title="InventoryIQ Dashboard",
    page_icon="📦",
    layout="wide"
)

# ===========================
# ESTILOS CSS PERSONALIZADOS
# ===========================
st.markdown("""
<style>
    /* Botón principal */
    .stButton > button {
        background: linear-gradient(90deg, #3b82f6, #2563eb);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #2563eb, #1d4ed8);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
        transform: translateY(-2px);
    }
    
    /* Header */
    h1 {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #f8fafc;
    }
    
    /* Métricas */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ===========================
# HEADER PRINCIPAL
# ===========================
st.title("📦 InventoryIQ Dashboard")
st.markdown("### Predicción de Demanda con Machine Learning")
st.markdown("---")

# ===========================
# SIDEBAR - INPUTS
# ===========================
st.sidebar.header("⚙️ Configuración de Predicción")
st.sidebar.markdown("Selecciona los parámetros para consultar el modelo XGBoost.")

# Selectores
min_date = datetime.date(2013, 1, 1)
max_date = datetime.date(2017, 12, 31)

selected_date = st.sidebar.date_input(
    "📅 Fecha",
    value=datetime.date(2017, 7, 15),
    min_value=min_date,
    max_value=max_date,
    help="Selecciona una fecha dentro del rango de datos históricos"
)

store_id = st.sidebar.number_input(
    "🏪 ID de Tienda",
    min_value=1,
    max_value=10,
    value=1,
    help="ID de la tienda (1-10)"
)

item_id = st.sidebar.number_input(
    "📦 ID de Producto",
    min_value=1,
    max_value=50,
    value=1,
    help="ID del producto (1-50)"
)

st.sidebar.markdown("---")

# Botón de Predicción
predict_button = st.sidebar.button("🔮 Generar Predicción", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("InventoryIQ v2.0 | Powered by XGBoost & FastAPI")

# ===========================
# LÓGICA DE PREDICCIÓN
# ===========================
if predict_button:
    # URL configurable via variable de entorno
    API_BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
    api_url = f"{API_BASE_URL}/predict"
    
    payload = {
        "date": str(selected_date),
        "store": store_id,
        "item": item_id
    }
    
    with st.spinner('🧠 Consultando modelo de IA...'):
        try:
            response = requests.post(api_url, json=payload, timeout=5)
            
            if response.status_code == 200:
                result = response.json()
                prediction = result['prediction']
                
                # ===========================
                # RESULTADOS - 4 COLUMNAS CON MÉTRICAS
                # ===========================
                st.success("✅ Predicción Generada Exitosamente")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        label="📅 Fecha",
                        value=str(selected_date)
                    )
                
                with col2:
                    st.metric(
                        label="🏪 Tienda",
                        value=f"#{store_id}"
                    )
                
                with col3:
                    st.metric(
                        label="📦 Producto",
                        value=f"#{item_id}"
                    )
                
                with col4:
                    st.metric(
                        label="🎯 Predicción",
                        value=f"{prediction:.0f}",
                        delta="Unidades"
                    )
                
                st.markdown("---")
                
                # ===========================
                # GRÁFICO GAUGE (TACÓMETRO)
                # ===========================
                st.subheader("📊 Visualización de Demanda Esperada")
                st.caption("El indicador muestra la magnitud de la predicción en escala visual")
                
                # Rango dinámico (asumimos un máximo de 100 para fines visuales)
                max_range = max(100, prediction * 1.5)
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prediction,
                    title={'text': f"<b>Unidades Predichas</b><br><span style='font-size:0.8em; color:gray'>Rango: 0-{max_range:.0f}</span>", 'font': {'size': 20}},
                    number={'suffix': " unidades", 'font': {'size': 40}},
                    gauge={
                        'axis': {
                            'range': [None, max_range], 
                            'tickwidth': 2, 
                            'tickcolor': "#64748b",
                            'ticksuffix': ' u'
                        },
                        'bar': {'color': "#3b82f6", 'thickness': 0.75},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "#e2e8f0",
                        'steps': [
                            {'range': [0, max_range * 0.33], 'color': '#dbeafe', 'name': 'Baja'},
                            {'range': [max_range * 0.33, max_range * 0.66], 'color': '#93c5fd', 'name': 'Media'},
                            {'range': [max_range * 0.66, max_range], 'color': '#60a5fa', 'name': 'Alta'}
                        ],
                        'threshold': {
                            'line': {'color': "#dc2626", 'width': 4},
                            'thickness': 0.75,
                            'value': max_range * 0.9
                        }
                    }
                ))
                
                fig.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=80, b=20),
                    paper_bgcolor="#ffffff",
                    font={'color': "#1e293b", 'family': "Arial"}
                )
                
                
                # Leyenda explicativa con mejor diseño y valores concretos
                st.markdown("##### 📋 Guía de Interpretación")
                
                # Calcular rangos concretos
                low_max = int(max_range * 0.33)
                med_min = low_max
                med_max = int(max_range * 0.66)
                high_min = med_max
                critical_threshold = int(max_range * 0.9)
                
                col_leg1, col_leg2 = st.columns(2)
                
                with col_leg1:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); 
                                padding: 15px; border-radius: 10px; border-left: 4px solid #3b82f6;'>
                        <p style='margin: 0; color: #1e40af; font-weight: 600; font-size: 14px;'>
                            🔵 Demanda Baja
                        </p>
                        <p style='margin: 5px 0 0 0; color: #1e3a8a; font-size: 13px;'>
                            <b>0 - {low_max} unidades</b>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%); 
                                padding: 15px; border-radius: 10px; border-left: 4px solid #2563eb; margin-top: 10px;'>
                        <p style='margin: 0; color: white; font-weight: 600; font-size: 14px;'>
                            🔷 Demanda Alta
                        </p>
                        <p style='margin: 5px 0 0 0; color: #e0f2fe; font-size: 13px;'>
                            <b>{high_min} - {int(max_range)} unidades</b>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_leg2:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #93c5fd 0%, #60a5fa 100%); 
                                padding: 15px; border-radius: 10px; border-left: 4px solid #3b82f6;'>
                        <p style='margin: 0; color: #1e3a8a; font-weight: 600; font-size: 14px;'>
                            🔹 Demanda Media
                        </p>
                        <p style='margin: 5px 0 0 0; color: #1e40af; font-size: 13px;'>
                            <b>{med_min} - {med_max} unidades</b>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #fecaca 0%, #f87171 100%); 
                                padding: 15px; border-radius: 10px; border-left: 4px solid #dc2626; margin-top: 10px;'>
                        <p style='margin: 0; color: #7f1d1d; font-weight: 600; font-size: 14px;'>
                            🔴 Umbral Crítico
                        </p>
                        <p style='margin: 5px 0 0 0; color: #991b1b; font-size: 13px;'>
                            <b>A partir de {critical_threshold} unidades</b>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # ===========================
                # GRÁFICO DE HISTORIAL
                # ===========================
                st.markdown("---")
                st.subheader("📈 Historial de Ventas")
                
                try:
                    # Llamar al endpoint /history
                    history_url = f"{API_BASE_URL}/history"
                    hist_response = requests.get(
                        history_url, 
                        params={"store": store_id, "item": item_id, "limit": 30},
                        timeout=5
                    )
                    
                    if hist_response.status_code == 200:
                        history_data = hist_response.json()
                        
                        if history_data:
                            hist_df = pd.DataFrame(history_data)
                            hist_df = hist_df.sort_values('date')  # Ordenar ascendente para el gráfico
                            
                            # Crear gráfico de línea
                            import plotly.express as px
                            fig_history = px.line(
                                hist_df, 
                                x='date', 
                                y='sales',
                                title=f'Últimas 30 Ventas - Tienda {store_id}, Producto {item_id}',
                                labels={'date': 'Fecha', 'sales': 'Unidades Vendidas'},
                                markers=True
                            )
                            
                            fig_history.update_traces(
                                line_color='#3b82f6',
                                line_width=3,
                                marker=dict(size=8, color='#2563eb')
                            )
                            
                            fig_history.update_layout(
                                height=350,
                                margin=dict(l=20, r=20, t=60, b=20),
                                plot_bgcolor='#f8fafc',
                                paper_bgcolor='#ffffff',
                                font={'color': "#000000", 'family': "Arial", 'size': 12},
                                hovermode='x unified',
                                xaxis=dict(
                                    title_font=dict(color='#000000', size=14),
                                    tickfont=dict(color='#000000', size=12)
                                ),
                                yaxis=dict(
                                    title_font=dict(color='#000000', size=14),
                                    tickfont=dict(color='#000000', size=12)
                                )
                            )
                            
                            st.plotly_chart(fig_history, use_container_width=True)
                            
                            # Métricas de contexto
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("📊 Promedio Histórico", f"{hist_df['sales'].mean():.1f}")
                            with col_b:
                                st.metric("📈 Máximo", f"{hist_df['sales'].max():.0f}")
                            with col_c:
                                st.metric("📉 Mínimo", f"{hist_df['sales'].min():.0f}")
                        else:
                            st.warning("No hay suficiente historial para este producto/tienda.")
                    else:
                        st.warning("No se pudo cargar el historial de ventas.")
                        
                except Exception as e:
                    st.warning(f"Error al cargar historial: {e}")
                
                # ===========================
                # RECOMENDACIÓN
                # ===========================
                st.info(f"""
                **💡 Recomendación:**  
                Asegúrate de tener al menos **{prediction:.0f} unidades** en stock para el {selected_date}.  
                Considera un margen de seguridad del 10-15% para evitar roturas de inventario.
                """)
                
            elif response.status_code == 404:
                st.error("❌ No hay datos históricos disponibles para esta combinación de fecha/tienda/producto.")
            else:
                st.error(f"❌ Error del servidor: {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("""
            🔌 **No se pudo conectar con la API**  
            Asegúrate de que el servidor FastAPI esté ejecutándose:
            ```bash
            uvicorn src.api.main:app --reload
            ```
            """)
        except requests.exceptions.Timeout:
            st.error("⏱️ La solicitud excedió el tiempo límite. Intenta nuevamente.")
        except Exception as e:
            st.error(f"❌ Error inesperado: {e}")

else:
    # ===========================
    # ESTADO INICIAL
    # ===========================
    st.info("👈 **Configura los parámetros** en el menú lateral y presiona el botón para generar una predicción.")
    
    # Placeholder visual
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image("https://via.placeholder.com/300x200/3b82f6/ffffff?text=Machine+Learning", use_container_width=True)
    
    with col2:
        st.image("https://via.placeholder.com/300x200/8b5cf6/ffffff?text=XGBoost+Model", use_container_width=True)
    
    with col3:
        st.image("https://via.placeholder.com/300x200/10b981/ffffff?text=Real-Time+API", use_container_width=True)