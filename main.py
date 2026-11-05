import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# Modelos Tabulares
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

# Modelos de Series de Tiempo
import statsmodels.api as sm
from prophet import Prophet

# ============================================================
# CONFIGURACIÓN BÁSICA
# ============================================================
st.set_page_config(page_title="Dashboard Predicción Turismo Colombia", layout="wide")
st.title("Dashboard Multimodelo de Proyección Turística Colombia")

# ============================================================
# 1. CARGA DE DATOS DESDE LA API (CON PAGINACIÓN)
# ============================================================
st.sidebar.header("1. Carga de Datos")

@st.cache_data(ttl=3600, show_spinner="Descargando datos de turismo...")
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
                st.error(f"Error de conexión con la API de Turismo: Código {response.status_code}")
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

@st.cache_data(ttl=3600, show_spinner="Descargando datos históricos de la TRM...")
def load_trm_data():
    base_url = "https://www.datos.gov.co/resource/32sa-wavu.json?$limit=50000"
    try:
        response = requests.get(base_url, timeout=30)
        if response.status_code == 200:
            df_trm = pd.DataFrame(response.json())
            if not df_trm.empty and "valor" in df_trm.columns and "vigenciadesde" in df_trm.columns:
                df_trm["valor"] = pd.to_numeric(df_trm["valor"], errors="coerce")
                df_trm["fecha"] = pd.to_datetime(df_trm["vigenciadesde"], errors="coerce")
                df_trm = df_trm.dropna(subset=["valor", "fecha"])
                
                df_trm["a_o"] = df_trm["fecha"].dt.year
                df_trm["mes_num"] = df_trm["fecha"].dt.month
                
                trm_mensual = df_trm.groupby(["a_o", "mes_num"])["valor"].mean().reset_index()
                trm_mensual = trm_mensual.rename(columns={"valor": "trm_promedio"})
                return trm_mensual
    except Exception as e:
        st.sidebar.warning(f"No se pudo cargar la TRM histórica: {e}")
    return pd.DataFrame()

records_limit = st.sidebar.number_input(
    "Límite de registros (0 para descargar TODO):", 
    min_value=0, value=0, step=10000,
    help="0 descarga la base completa."
)

limit_to_pass = None if records_limit == 0 else records_limit
df_turismo = load_and_clean_data(max_records=limit_to_pass)
df_trm = load_trm_data()

if df_turismo.empty:
    st.error("Error al cargar los datos.")
    st.stop()
else:
    st.sidebar.success(f"Se cargaron {len(df_turismo)} registros de turismo.")

# Mapeo de meses para cruce con TRM
meses_map = {
    'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6, 
    'julio': 7, 'agosto': 8, 'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12,
    'ene': 1, 'feb': 2, 'mar': 3, 'abr': 4, 'may': 5, 'jun': 6, 
    'jul': 7, 'ago': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dic': 12,
    '01': 1, '02': 2, '03': 3, '04': 4, '05': 5, '06': 6, '07': 7, '08': 8, '09': 9,
    '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, '12': 12
}

if "mes" in df_turismo.columns:
    df_turismo["mes_num"] = df_turismo["mes"].astype(str).str.lower().str.strip().map(meses_map)

# Cruce de DataFrames
if not df_trm.empty and "a_o" in df_turismo.columns and "mes_num" in df_turismo.columns:
    df = pd.merge(df_turismo, df_trm, on=["a_o", "mes_num"], how="left")
    df["trm_promedio"] = df["trm_promedio"].ffill().bfill()
else:
    df = df_turismo.copy()

# Función auxiliar para convertir DataFrames a CSV descargables
@st.cache_data
def convert_df_to_csv(df_to_convert):
    return df_to_convert.to_csv(index=True).encode('utf-8')

# ============================================================
# 2. FILTROS DE ANÁLISIS ESTRATÉGICO
# ============================================================
st.sidebar.header("2. Filtros de Análisis")
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

st.sidebar.info(f"Registros a analizar tras filtros: {len(df_filtrado)}")

# Botón de descarga en el sidebar para los datos filtrados actuales
st.sidebar.download_button(
    label="Exportar datos filtrados a CSV",
    data=convert_df_to_csv(df_filtrado),
    file_name="datos_turismo_colombia.csv",
    mime="text/csv",
    key="download_filtered_df"
)

if df_filtrado.empty:
    st.warning("Los filtros aplicados dejaron la base de datos sin registros.")
    st.stop()

# ============================================================
# CREACIÓN DE PESTAÑAS
# ============================================================
tab1, tab2 = st.tabs(["Enfoque Tabular (LightGBM vs XGBoost)", "Series de Tiempo (ARIMAX vs Prophet)"])

# ============================================================
# PESTAÑA 1: TABULAR (LIGHTGBM VS XGBOOST)
# ============================================================
with tab1:
    st.header("Comparativa de Modelos Tabulares: LightGBM vs. XGBoost")
    
    if "pipelines" not in st.session_state:
        st.session_state.pipelines = None
    if "evaluation_metrics" not in st.session_state:
        st.session_state.evaluation_metrics = None

    def reset_model_state():
        st.session_state.pipelines = None
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

    st.sidebar.header("3. Hiperparámetros Tabulares")
    with st.sidebar.form("tabular_form"):
        test_size = st.slider("Porcentaje de prueba", 0.1, 0.5, 0.2, 0.05)
        n_estimators = st.slider("Número de estimadores", 10, 500, 100, 10)
        learning_rate = st.selectbox("Tasa de aprendizaje", [0.01, 0.05, 0.1, 0.2])
        max_depth = st.slider("Profundidad máxima", 1, 30, 6)
        submit_train = st.form_submit_button("Entrenar Ambos Modelos")

    if submit_train or st.session_state.pipelines is None:
        with st.spinner("Entrenando LightGBM y XGBoost..."):
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
            
            # Instanciación de modelos
            if is_numeric_target:
                model_lgb = LGBMRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42, n_jobs=2)
                model_xgb = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42, n_jobs=2)
            else:
                model_lgb = LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42, n_jobs=2)
                model_xgb = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42, n_jobs=2)
            
            pipe_lgb = Pipeline([("preprocessor", preprocessor), ("model", model_lgb)])
            pipe_xgb = Pipeline([("preprocessor", preprocessor), ("model", model_xgb)])
            
            pipe_lgb.fit(X_train, y_train)
            pipe_xgb.fit(X_train, y_train)
            
            y_pred_lgb = pipe_lgb.predict(X_test)
            y_pred_xgb = pipe_xgb.predict(X_test)
            
            st.session_state.pipelines = {"lgb": pipe_lgb, "xgb": pipe_xgb}
            st.session_state.evaluation_metrics = {
                "y_test": y_test,
                "y_pred_lgb": y_pred_lgb,
                "y_pred_xgb": y_pred_xgb
            }
            st.success("¡Modelos LightGBM y XGBoost entrenados exitosamente!")

    if st.session_state.evaluation_metrics:
        y_test = st.session_state.evaluation_metrics["y_test"]
        y_pred_lgb = st.session_state.evaluation_metrics["y_pred_lgb"]
        y_pred_xgb = st.session_state.evaluation_metrics["y_pred_xgb"]

        st.subheader("Tabla Comparativa de Rendimiento")
        
        if is_numeric_target:
            metrics_df = pd.DataFrame({
                "Métrica": ["R² Score (Mayor es mejor)", "MAPE (Menor es mejor)", "RMSE (Menor es mejor)"],
                "LightGBM": [
                    f"{r2_score(y_test, y_pred_lgb):.4f}",
                    f"{mean_absolute_percentage_error(y_test, y_pred_lgb)*100:.2f}%",
                    f"{np.sqrt(mean_squared_error(y_test, y_pred_lgb)):.2f}"
                ],
                "XGBoost": [
                    f"{r2_score(y_test, y_pred_xgb):.4f}",
                    f"{mean_absolute_percentage_error(y_test, y_pred_xgb)*100:.2f}%",
                    f"{np.sqrt(mean_squared_error(y_test, y_pred_xgb)):.2f}"
                ]
            })
            st.dataframe(metrics_df, use_container_width=True)

            # Gráfico de dispersión comparativo
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            
            ax[0].scatter(y_test, y_pred_lgb, alpha=0.5, color="teal")
            ax[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
            ax[0].set(xlabel="Real", ylabel="Predicción", title="LightGBM: Real vs Predicho")

            ax[1].scatter(y_test, y_pred_xgb, alpha=0.5, color="coral")
            ax[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
            ax[1].set(xlabel="Real", ylabel="Predicción", title="XGBoost: Real vs Predicho")
            
            st.pyplot(fig)
            plt.close(fig)
        else:
            col_acc1, col_acc2 = st.columns(2)
            col_acc1.metric("Exactitud LightGBM", f"{accuracy_score(y_test, y_pred_lgb) * 100:.2f}%")
            col_acc2.metric("Exactitud XGBoost", f"{accuracy_score(y_test, y_pred_xgb) * 100:.2f}%")

# ============================================================
# PESTAÑA 2: SERIES DE TIEMPO (ARIMAX VS PROPHET CON TRM)
# ============================================================
with tab2:
    st.header("Análisis de Series de Tiempo: ARIMAX vs. Meta Prophet (con Variable Exógena TRM)")
    
    if "a_o" in df_filtrado.columns and "mes_num" in df_filtrado.columns and "cant_extranjeros_no_residentes" in df_filtrado.columns:
        ts_df = df_filtrado.dropna(subset=['a_o', 'mes_num']).copy()
        ts_df['fecha'] = pd.to_datetime(ts_df['a_o'].astype(int).astype(str) + '-' + ts_df['mes_num'].astype(int).astype(str) + '-01')
        
        agg_dict = {'cant_extranjeros_no_residentes': 'sum'}
        if 'trm_promedio' in ts_df.columns:
            agg_dict['trm_promedio'] = 'mean'
            
        ts_grouped = ts_df.groupby('fecha').agg(agg_dict).reset_index()
        ts_grouped = ts_grouped.sort_values('fecha').set_index('fecha')
        ts_grouped = ts_grouped.resample('MS').asfreq().ffill().bfill()
        
        st.subheader("Volumen Histórico de Turistas")
        st.line_chart(ts_grouped['cant_extranjeros_no_residentes'])

        if 'trm_promedio' in ts_grouped.columns:
            st.subheader("Evolución Histórica TRM (USD/COP)")
            st.line_chart(ts_grouped['trm_promedio'])

        st.download_button(
            label="Exportar datos de la serie de tiempo a CSV",
            data=convert_df_to_csv(ts_grouped),
            file_name="serie_tiempo_turismo_trm.csv",
            mime="text/csv",
            key="download_ts_grouped"
        )
        
        if len(ts_grouped) > 12:
            st.subheader("Proyección a 6 Meses con Rangos de Incertidumbre y TRM Exógena")
            
            with st.spinner("Ajustando ARIMAX y Prophet incorporando la TRM..."):
                exog_hist = ts_grouped[['trm_promedio']] if 'trm_promedio' in ts_grouped.columns else None
                
                # Asumimos la última TRM conocida para los próximos 6 meses
                if exog_hist is not None:
                    last_trm = exog_hist.iloc[-1, 0]
                    future_exog = pd.DataFrame({'trm_promedio': [last_trm] * 6}, 
                                               index=pd.date_range(start=ts_grouped.index[-1] + pd.DateOffset(months=1), periods=6, freq='MS'))
                else:
                    future_exog = None

                # 1. ENTRENAMIENTO ARIMAX
                mod_sarimax = sm.tsa.statespace.SARIMAX(
                    ts_grouped['cant_extranjeros_no_residentes'],
                    exog=exog_hist,
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 0, 12),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                res_sarimax = mod_sarimax.fit(disp=False)
                pred_sarimax = res_sarimax.get_forecast(steps=6, exog=future_exog)
                df_sarimax_pred = pred_sarimax.predicted_mean
                ci_sarimax = pred_sarimax.conf_int()

                # 2. ENTRENAMIENTO PROPHET
                df_prophet_in = ts_grouped.reset_index().rename(columns={'fecha': 'ds', 'cant_extranjeros_no_residentes': 'y'})
                m_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                
                if 'trm_promedio' in df_prophet_in.columns:
                    m_prophet.add_regressor('trm_promedio')
                    
                m_prophet.fit(df_prophet_in)
                
                future_dates = m_prophet.make_future_dataframe(periods=6, freq='MS')
                if 'trm_promedio' in df_prophet_in.columns:
                    # Rellenar exógena futura
                    future_dates['trm_promedio'] = list(df_prophet_in['trm_promedio']) + [last_trm] * 6
                    
                forecast_prophet = m_prophet.predict(future_dates)
                df_prophet_tail = forecast_prophet.tail(6).set_index('ds')
                df_prophet_pred = df_prophet_tail['yhat']
                ci_prophet_lower = df_prophet_tail['yhat_lower']
                ci_prophet_upper = df_prophet_tail['yhat_upper']

            # VISUALIZACIÓN GRÁFICA COMPARATIVA CON INTERVALOS DE CONFIANZA
            fig_ts, ax_ts = plt.subplots(figsize=(12, 6))
            ax_ts.plot(ts_grouped.index, ts_grouped['cant_extranjeros_no_residentes'], label='Histórico Real', color='black', lw=1.8)
            
            # ARIMAX
            ax_ts.plot(df_sarimax_pred.index, df_sarimax_pred, color='red', linestyle='--', marker='o', label='Proyección ARIMAX')
            ax_ts.fill_between(
                ci_sarimax.index, 
                ci_sarimax.iloc[:, 0].clip(lower=0), 
                ci_sarimax.iloc[:, 1], 
                color='red', alpha=0.15, label='Incertidumbre ARIMAX (Pesimista / Optimista)'
            )

            # PROPHET
            ax_ts.plot(df_prophet_pred.index, df_prophet_pred, color='blue', linestyle='--', marker='s', label='Proyección Prophet')
            ax_ts.fill_between(
                df_prophet_pred.index, 
                ci_prophet_lower.clip(lower=0), 
                ci_prophet_upper, 
                color='blue', alpha=0.15, label='Incertidumbre Prophet (Pesimista / Optimista)'
            )
            
            ax_ts.legend(loc='upper left')
            ax_ts.set_title("Proyección Multimodelo de Demanda Turística en Colombia")
            ax_ts.set_ylabel("Cantidad de Turistas")
            st.pyplot(fig_ts)
            plt.close(fig_ts)
            
            # TABLA COMPARATIVA DE VALORES PREDICHOS Y ESCENARIOS
            st.subheader("Valores Numéricos y Escenarios por Modelo")
            df_comp_pred = pd.DataFrame({
                'Mes Proyectado': df_sarimax_pred.index.strftime('%Y-%m'),
                'ARIMAX - Pesimista': ci_sarimax.iloc[:, 0].clip(lower=0).round(0).values,
                'ARIMAX - Esperado': df_sarimax_pred.values.round(0),
                'ARIMAX - Optimista': ci_sarimax.iloc[:, 1].round(0).values,
                'Prophet - Pesimista': ci_prophet_lower.clip(lower=0).round(0).values,
                'Prophet - Esperado': df_prophet_pred.values.round(0),
                'Prophet - Optimista': ci_prophet_upper.round(0).values,
            }).set_index('Mes Proyectado')
            
            st.dataframe(df_comp_pred, use_container_width=True)

            st.download_button(
                label="Exportar tabla de predicciones a CSV",
                data=convert_df_to_csv(df_comp_pred),
                file_name="proyecciones_arimax_vs_prophet_trm.csv",
                mime="text/csv",
                key="download_predictions"
            )
            
        else:
            st.warning(f"Se requieren al menos 24 meses continuos de datos. Se detectaron {len(ts_grouped)} meses.")
    else:
        st.error("No se encontraron las columnas necesarias ('a_o', 'mes', 'cant_extranjeros_no_residentes').")