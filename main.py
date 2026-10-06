import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from lightgbm import LGBMClassifier, LGBMRegressor
import statsmodels.api as sm

# ============================================================
# CONFIGURACIÓN BÁSICA
# ============================================================
st.set_page_config(page_title="Dashboard Predicción Turismo - LightGBM & ARIMAX", layout="wide")
st.title("📊 Dashboard de Proyección Turística Colombia (LightGBM & ARIMAX)")

# ============================================================
# 1. CARGA DE DATOS DESDE LA API (CON PAGINACIÓN)
# ============================================================
st.sidebar.header("1. Carga de Datos")

@st.cache_data(ttl=3600, show_spinner="Descargando datos en bloques (esto puede tomar un par de minutos)...")
def load_and_clean_data(max_records=None):
    base_url = "https://www.datos.gov.co/resource/7wm8-w5ad.json"
    all_data = []
    offset = 0
    limit_per_request = 50000
    
    try:
        while True:
            url = f"{base_url}?$limit={limit_per_request}&$offset={offset}"
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                chunk = response.json()
                if not chunk:
                    break
                all_data.extend(chunk)
                offset += limit_per_request
                
                if max_records and len(all_data) >= max_records:
                    all_data = all_data[:max_records]
                    break
            else:
                st.error(f"Error de conexión con la API: Código {response.status_code}")
                break
                
        df = pd.DataFrame(all_data)
        
        if df.empty:
            return df
            
        if "a_o" in df.columns:
            df["a_o"] = pd.to_numeric(df["a_o"], errors="coerce")
        if "cant_extranjeros_no_residentes" in df.columns:
            df["cant_extranjeros_no_residentes"] = pd.to_numeric(df["cant_extranjeros_no_residentes"], errors="coerce")
            
        if "mes" in df.columns:
            df["mes"] = df["mes"].astype(str).str.strip()
            
        if "cant_extranjeros_no_residentes" in df.columns:
            df = df.dropna(subset=["cant_extranjeros_no_residentes"])
            
        df = df.fillna("Desconocido")
        return df
        
    except requests.exceptions.RequestException as e:
        st.error(f"El servidor tardó demasiado o rechazó la conexión: {e}")
        return pd.DataFrame()

records_limit = st.sidebar.number_input(
    "Límite de registros (0 para descargar TODO):", 
    min_value=0, value=0, step=10000,
    help="Pon 0 para descargar la base de datos completa. Atención: descargar todos los datos tomará más tiempo."
)

limit_to_pass = None if records_limit == 0 else records_limit
df = load_and_clean_data(max_records=limit_to_pass)

if df.empty:
    st.error("Error al cargar los datos.")
    st.stop()
else:
    st.sidebar.success(f"✅ Se cargaron {len(df)} registros.")

# ============================================================
# 2. FILTROS DE ANÁLISIS ESTRATÉGICO
# ============================================================
st.sidebar.header("2. Filtros de Análisis")
st.sidebar.write("Deja en blanco para incluir todos.")

col_depto = next((c for c in df.columns if 'departamento' in c.lower()), None)
col_ciudad = next((c for c in df.columns if 'ciudad' in c.lower()), None)
col_pais = next((c for c in df.columns if 'pais' in c.lower() or 'residencia' in c.lower()), None)

def crear_filtro(columna, titulo):
    if columna:
        opciones = df[columna].astype(str).unique().tolist()
        opciones.sort()
        return st.sidebar.multiselect(f"{titulo}:", options=opciones, default=[])
    return []

filtro_depto = crear_filtro(col_depto, "Departamento")
filtro_ciudad = crear_filtro(col_ciudad, "Ciudad")
filtro_pais = crear_filtro(col_pais, "País de Residencia")

df_filtrado = df.copy()

if col_depto and len(filtro_depto) > 0:
    df_filtrado = df_filtrado[df_filtrado[col_depto].isin(filtro_depto)]
if col_ciudad and len(filtro_ciudad) > 0:
    df_filtrado = df_filtrado[df_filtrado[col_ciudad].isin(filtro_ciudad)]
if col_pais and len(filtro_pais) > 0:
    df_filtrado = df_filtrado[df_filtrado[col_pais].isin(filtro_pais)]

st.sidebar.info(f"📊 Registros a analizar tras filtros: {len(df_filtrado)}")

if df_filtrado.empty:
    st.warning("⚠️ Los filtros aplicados dejaron la base de datos sin registros. Por favor, amplía tu selección.")
    st.stop()

# ============================================================
# CREACIÓN DE PESTAÑAS
# ============================================================
tab1, tab2 = st.tabs(["🚀 Modelo LightGBM (Tabular)", "📈 Modelo ARIMAX (Series de Tiempo)"])

# ============================================================
# PESTAÑA 1: LIGHTGBM
# ============================================================
with tab1:
    st.header("Enfoque Tabular con LightGBM")
    
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "evaluation_metrics" not in st.session_state:
        st.session_state.evaluation_metrics = None

    def reset_model_state():
        st.session_state.pipeline = None
        st.session_state.evaluation_metrics = None

    target_col = st.selectbox(
        "Selecciona la columna objetivo (target):",
        options=df_filtrado.columns,
        index=list(df_filtrado.columns).index("cant_extranjeros_no_residentes") if "cant_extranjeros_no_residentes" in df_filtrado.columns else 0,
        on_change=reset_model_state
    )

    is_numeric_target = pd.api.types.is_numeric_dtype(df_filtrado[target_col]) and df_filtrado[target_col].nunique() > 20

    X = df_filtrado.drop(columns=[target_col])
    y = df_filtrado[target_col] if is_numeric_target else df_filtrado[target_col].astype(str)

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    st.sidebar.header("3. Hiperparámetros LightGBM")
    with st.sidebar.form("lgbm_form"):
        test_size = st.slider("Porcentaje de prueba", 0.1, 0.5, 0.2, 0.05)
        n_estimators = st.slider("Número de estimadores", 10, 500, 100, 10)
        learning_rate = st.selectbox("Tasa de aprendizaje", [0.01, 0.05, 0.1, 0.2])
        max_depth = st.slider("Profundidad máxima", -1, 30, -1)
        submit_train = st.form_submit_button("Entrenar LightGBM")

    if submit_train or st.session_state.pipeline is None:
        with st.spinner("Entrenando LightGBM..."):
            stratify_value = y if (not is_numeric_target and y.nunique() < 20 and y.value_counts().min() >= 2) else None
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=stratify_value
            )
            
            transformers = []
            if numeric_features:
                transformers.append(("num", "passthrough", numeric_features))
            if categorical_features:
                transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features))
                
            preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
            
            ModelClass = LGBMRegressor if is_numeric_target else LGBMClassifier
            model = ModelClass(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42, n_jobs=-1)
            
            pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
            pipeline.fit(X_train, y_train)
            
            y_pred = pipeline.predict(X_test)
            
            st.session_state.pipeline = pipeline
            st.session_state.evaluation_metrics = {"y_test": y_test, "y_pred": y_pred}
            st.success("¡Modelo LightGBM entrenado con los datos filtrados!")

    if st.session_state.evaluation_metrics:
        y_test = st.session_state.evaluation_metrics["y_test"]
        y_pred = st.session_state.evaluation_metrics["y_pred"]

        col1, col2 = st.columns([1, 2])
        with col1:
            if is_numeric_target:
                st.metric("MAPE (Error Absoluto Porcentual)", f"{mean_absolute_percentage_error(y_test, y_pred)*100:.2f}%")
                st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
                st.metric("R² Score", f"{r2_score(y_test, y_pred):.4f}")
            else:
                st.metric("Exactitud (Accuracy)", f"{accuracy_score(y_test, y_pred) * 100:.2f}%")
                st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True, zero_division=0)).T.style.format(precision=2))

        with col2:
            if is_numeric_target:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(y_test, y_pred, alpha=0.5, color="teal")
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
                ax.set(xlabel="Real", ylabel="Predicción (LightGBM)", title="Valores Reales vs Predichos")
                st.pyplot(fig)
                plt.close(fig)
            else:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set(xlabel="Predicción", ylabel="Valor Real", title="Matriz de Confusión")
                st.pyplot(fig)
                plt.close(fig)

# ============================================================
# PESTAÑA 2: ARIMAX (SERIES DE TIEMPO)
# ============================================================
with tab2:
    st.header("Análisis de Series de Tiempo (ARIMAX)")
    st.write("Agrupación del volumen histórico por Año y Mes para la predicción estacional.")
    
    if "a_o" in df_filtrado.columns and "mes" in df_filtrado.columns and "cant_extranjeros_no_residentes" in df_filtrado.columns:
        ts_df = df_filtrado.copy()
        ts_df['mes_limpio'] = ts_df['mes'].astype(str).str.lower().str.strip()
        
        meses_map = {
            'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6, 
            'julio': 7, 'agosto': 8, 'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12,
            'ene': 1, 'feb': 2, 'mar': 3, 'abr': 4, 'may': 5, 'jun': 6, 
            'jul': 7, 'ago': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dic': 12,
            '01': 1, '02': 2, '03': 3, '04': 4, '05': 5, '06': 6, '07': 7, '08': 8, '09': 9,
            '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, '12': 12
        }
        
        ts_df['mes_num'] = ts_df['mes_limpio'].map(meses_map)
        ts_df = ts_df.dropna(subset=['a_o', 'mes_num'])
        ts_df['fecha'] = pd.to_datetime(ts_df['a_o'].astype(int).astype(str) + '-' + ts_df['mes_num'].astype(int).astype(str) + '-01')
        
        ts_grouped = ts_df.groupby('fecha')['cant_extranjeros_no_residentes'].sum().reset_index()
        ts_grouped = ts_grouped.sort_values('fecha').set_index('fecha')
        
        # Corrección: Forzar índice mensual continuo y rellenar vacíos con 0
        ts_grouped = ts_grouped.resample('MS').asfreq().fillna(0)
        
        st.line_chart(ts_grouped['cant_extranjeros_no_residentes'])
        
        with st.expander("Ver tabla detallada de datos históricos (Mes a Mes)"):
            df_hist_display = ts_grouped.copy()
            df_hist_display.index = df_hist_display.index.strftime('%Y-%m')
            df_hist_display = df_hist_display.rename(columns={'cant_extranjeros_no_residentes': 'Volumen de Turistas'})
            st.dataframe(df_hist_display, use_container_width=True)
        
        if len(ts_grouped) > 12:
            st.subheader("Predicción Estacional (SARIMAX)")
            
            p, d, q = 1, 1, 1
            P, D, Q, s = 1, 1, 0, 12
            
            try:
                with st.spinner("Ajustando modelo ARIMAX... esto puede tomar unos segundos"):
                    mod = sm.tsa.statespace.SARIMAX(ts_grouped['cant_extranjeros_no_residentes'],
                                                    order=(p, d, q),
                                                    seasonal_order=(P, D, Q, s),
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False)
                    results = mod.fit(disp=False)
                    
                    pred = results.get_forecast(steps=6)
                    pred_ci = pred.conf_int()
                    
                    fig2, ax2 = plt.subplots(figsize=(10, 5))
                    ax2.plot(ts_grouped.index, ts_grouped['cant_extranjeros_no_residentes'], label='Histórico Filtrado')
                    ax2.plot(pred.predicted_mean.index, pred.predicted_mean, color='red', label='Proyección (Próximos 6 meses)')
                    ax2.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='pink', alpha=0.3, label='Intervalo de Confianza')
                    ax2.legend()
                    ax2.set_title("Proyección de Demanda Turística Segmentada")
                    st.pyplot(fig2)
                    
                    with st.expander("Ver tabla numérica de predicciones exactas"):
                        pred_df = pd.DataFrame({
                            'Mes Proyectado': pred.predicted_mean.index.strftime('%Y-%m'),
                            'Proyección Esperada': pred.predicted_mean.round(0),
                            'Escenario Pesimista (Límite Inferior)': pred_ci.iloc[:, 0].clip(lower=0).round(0),
                            'Escenario Optimista (Límite Superior)': pred_ci.iloc[:, 1].round(0)
                        }).set_index('Mes Proyectado')
                        st.dataframe(pred_df, use_container_width=True)
                        
            except Exception as e:
                st.error(f"Error al calcular ARIMAX con los datos actuales: {e}")
        else:
            st.warning(f"Apenas se detectaron {len(ts_grouped)} meses de datos válidos tras aplicar los filtros. Se requieren al menos 24 meses continuos para la predicción.")
    else:
        st.error("No se encontraron las columnas necesarias ('a_o', 'mes', 'cant_extranjeros_no_residentes').")