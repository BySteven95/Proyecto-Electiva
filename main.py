import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    r2_score
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# ============================================================
# CONFIGURACIÓN BÁSICA
# ============================================================
st.set_page_config(page_title="Dashboard Random Forest - Datos Abiertos", layout="wide")
st.title("🌲 Dashboard de Evaluación y Proyección: Turismo Colombia")

# ============================================================
# 1. CARGA DE DATOS DESDE LA API
# ============================================================
st.sidebar.header("1. Carga de Datos")

@st.cache_data(ttl=3600, show_spinner="Descargando datos...")
def load_and_clean_data(limit=10000):
    url = f"https://www.datos.gov.co/resource/7wm8-w5ad.json?$limit={limit}"
    
    try:
        # Añadimos un timeout para evitar que se quede colgado
        response = requests.get(url, timeout=20)
        
        if response.status_code == 200:
            df = pd.DataFrame(response.json())
            
            if df.empty:
                st.error("La API devolvió un JSON vacío.")
                return df
                
            # Convertir columnas sin borrar nulos masivamente todavía
            if "a_o" in df.columns:
                df["a_o"] = pd.to_numeric(df["a_o"], errors="coerce")
            if "cant_extranjeros_no_residentes" in df.columns:
                df["cant_extranjeros_no_residentes"] = pd.to_numeric(df["cant_extranjeros_no_residentes"], errors="coerce")
                
            if "mes" in df.columns:
                df["mes"] = df["mes"].astype(str).str.strip()
                
            # Eliminar nulos SOLO en la variable objetivo para no perder todo el dataset
            if "cant_extranjeros_no_residentes" in df.columns:
                df = df.dropna(subset=["cant_extranjeros_no_residentes"])
                
            # Para el resto de columnas, rellenamos los nulos categóricos con "Desconocido"
            df = df.fillna("Desconocido")
            
            return df
            
        else:
            st.error(f"Error de conexión con la API: Código {response.status_code}")
            return pd.DataFrame()
            
    except requests.exceptions.RequestException as e:
        st.error(f"El servidor tardó demasiado o rechazó la conexión: {e}")
        return pd.DataFrame()

records_limit = st.sidebar.number_input("Límite de registros a descargar:", min_value=500, max_value=50000, value=10000, step=500)
df = load_and_clean_data(limit=records_limit)

if df.empty:
    st.error("Error al cargar los datos.")
    st.stop()

# ============================================================
# INICIALIZACIÓN DEL ESTADO DE SESIÓN
# ============================================================
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "evaluation_metrics" not in st.session_state:
    st.session_state.evaluation_metrics = None

# Función para borrar el modelo guardado si el usuario cambia el Target
def reset_model_state():
    st.session_state.pipeline = None
    st.session_state.evaluation_metrics = None

# Selección de Variable Objetivo (Añadimos el evento on_change)
target_col = st.sidebar.selectbox(
    "Selecciona la columna objetivo (target):",
    options=df.columns,
    index=list(df.columns).index("cant_extranjeros_no_residentes") if "cant_extranjeros_no_residentes" in df.columns else 0,
    on_change=reset_model_state
)

is_numeric_target = pd.api.types.is_numeric_dtype(df[target_col]) and df[target_col].nunique() > 20

# División de características
X = df.drop(columns=[target_col])

# Prevención de error de Sklearn: Forzar a string si es clasificación
if not is_numeric_target:
    y = df[target_col].astype(str)
else:
    y = df[target_col]

numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

# ============================================================
# ENTRENAMIENTO DEL MODELO (MEDIANTE FORMULARIO)
# ============================================================
st.sidebar.header("2. Configuración y Entrenamiento")

with st.sidebar.form("hyperparameters_form"):
    test_size = st.slider("Porcentaje de prueba", 0.1, 0.5, 0.2, 0.05)
    n_estimators = st.slider("Número de árboles", 10, 500, 100, 10)
    max_depth_val = st.slider("Profundidad máxima", 1, 30, 10)
    
    crit_options = ["squared_error", "absolute_error", "friedman_mse"] if is_numeric_target else ["gini", "entropy", "log_loss"]
    criterion = st.selectbox("Criterio de división", crit_options)
    random_state = st.number_input("Semilla aleatoria", value=42, step=1)
    
    submit_train = st.form_submit_button("Entrenar Modelo")

if submit_train or st.session_state.pipeline is None:
    with st.spinner("Entrenando modelo..."):
        stratify_value = y if (not is_numeric_target and y.nunique() < 20 and y.value_counts().min() >= 2) else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_value
        )
        
        transformers = []
        if numeric_features:
            transformers.append(("num", "passthrough", numeric_features))
        if categorical_features:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features))
            
        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
        
        ModelClass = RandomForestRegressor if is_numeric_target else RandomForestClassifier
        model = ModelClass(n_estimators=n_estimators, max_depth=max_depth_val, criterion=criterion, random_state=random_state, n_jobs=-1)
        
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        
        st.session_state.pipeline = pipeline
        st.session_state.evaluation_metrics = {"y_test": y_test, "y_pred": y_pred}
        st.sidebar.success("¡Modelo entrenado!")

# ============================================================
# 3. MÉTRICAS Y DESEMPEÑO
# ============================================================
st.subheader("3. Métricas y Desempeño del Modelo")
col1, col2 = st.columns([1, 2])
y_test = st.session_state.evaluation_metrics["y_test"]
y_pred = st.session_state.evaluation_metrics["y_pred"]

with col1:
    if not is_numeric_target:
        st.metric("Exactitud (Accuracy)", f"{accuracy_score(y_test, y_pred) * 100:.2f}%")
        st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True, zero_division=0)).T.style.format(precision=2))
    else:
        st.metric("R² Score", f"{r2_score(y_test, y_pred):.4f}")
        st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

with col2:
    if not is_numeric_target:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set(xlabel="Predicción", ylabel="Valor Real", title="Matriz de Confusión")
        st.pyplot(fig)
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(y_test, y_pred, alpha=0.5, color="teal")
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        ax.set(xlabel="Real", ylabel="Predicción", title="Valores Reales vs Predichos")
        st.pyplot(fig)
        plt.close(fig)

# ============================================================
# 4. PREDICCIÓN PUNTUAL RAPIDA
# ============================================================
st.divider()
col3, col4 = st.columns([2, 1])

with col3:
    st.subheader("Importancia de Variables")
    fitted_preprocessor = st.session_state.pipeline.named_steps["preprocessor"]
    importances = st.session_state.pipeline.named_steps["model"].feature_importances_
    
    feat_df = pd.DataFrame({
        "Feature": [f.replace("num__", "").replace("cat__", "") for f in fitted_preprocessor.get_feature_names_out()],
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(10)
    
    fig_feat, ax_feat = plt.subplots(figsize=(8, 4))
    sns.barplot(x="Importance", y="Feature", data=feat_df, palette="viridis", ax=ax_feat)
    st.pyplot(fig_feat)
    plt.close(fig_feat)

with col4:
    st.subheader("Predicción Puntual")
    with st.form("prediction_form"):
        user_inputs = {}
        for feat in feat_df["Feature"].head(3):
            orig_col = next((c for c in X.columns if feat == c or feat.startswith(c + "_")), None)
            if orig_col and orig_col not in user_inputs:
                if orig_col in numeric_features:
                    user_inputs[orig_col] = st.number_input(orig_col, value=float(X[orig_col].mean()))
                else:
                    user_inputs[orig_col] = st.selectbox(orig_col, X[orig_col].unique().tolist())
        
        predict_btn = st.form_submit_button("Ejecutar Predicción")

    if predict_btn:
        input_data = {}
        for col in X.columns:
            if col in user_inputs:
                input_data[col] = [user_inputs[col]]
            else:
                input_data[col] = [X[col].mean() if col in numeric_features else X[col].mode()[0]]
                
        pred = st.session_state.pipeline.predict(pd.DataFrame(input_data))[0]
        st.success(f"Resultado Predicho: **{pred:.2f}**" if is_numeric_target else f"Resultado Predicho: **{pred}**")