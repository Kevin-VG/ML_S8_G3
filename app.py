"""Aplicación Streamlit - Parte 2 del trabajo aplicativo, Sesión 08."""
from pathlib import Path

import pandas as pd
import streamlit as st

from ml_caso_a import (
    COLUMNAS_CATEGORICAS,
    COLUMNAS_NUMERICAS,
    SEMILLA,
    cargar_datos,
    entrenar_modelos,
    validar_datos,
)

st.set_page_config(
    page_title="Riesgo académico | Parte 2",
    page_icon="🎓",
    layout="wide",
)

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1180px;}
.subtitulo {color: #a8c7e5; margin-top: -8px; margin-bottom: 22px;}
.indicacion {border-left: 3px solid #2998ff; padding-left: 14px; margin: 18px 0;}
</style>
""", unsafe_allow_html=True)

st.title("🎓 Identificación temprana de riesgo académico")
st.markdown(
    '<p class="subtitulo">Sesión 08 · Caso A · Parte 2: partición y primeros modelos</p>',
    unsafe_allow_html=True,
)

st.sidebar.header("Datos del proyecto")
st.sidebar.caption("Por defecto se utiliza el archivo sintético proporcionado para la práctica.")
archivo_cargado = st.sidebar.file_uploader("Usar otro CSV con las mismas columnas (opcional)", type="csv")

try:
    if archivo_cargado is not None:
        datos = pd.read_csv(archivo_cargado)
        validar_datos(datos)
    else:
        ruta_csv = Path(__file__).resolve().parent / "estudiantes_sinteticos_s08.csv"
        datos = cargar_datos(ruta_csv)
    X_train, X_test, y_train, y_test, modelos = entrenar_modelos(datos)
except (FileNotFoundError, ValueError, KeyError, pd.errors.ParserError) as error:
    st.error(f"No se pudo iniciar el proyecto: {error}")
    st.stop()

st.sidebar.success("Datos cargados y modelos entrenados")
st.sidebar.caption("Semilla: 42 · Train/test: 80/20 · Stratify: aprobo")

res1, res2, res3, res4 = st.columns(4)
res1.metric("Estudiantes", f"{len(datos):,}")
res2.metric("Entrenamiento", f"{len(X_train):,}")
res3.metric("Prueba", f"{len(X_test):,}")
res4.metric("Modelos iniciales", str(len(modelos)))

pestana_datos, pestana_preparacion, pestana_modelos, pestana_prediccion = st.tabs([
    "01 · Datos y partición",
    "02 · Preprocesamiento",
    "03 · Modelos entrenados",
    "04 · Predicción de ejemplo",
])

with pestana_datos:
    st.subheader("División 80% entrenamiento / 20% prueba")
    st.write(
        "La variable a predecir es **aprobo**: **0 = desaprobó (riesgo)** y "
        "**1 = aprobó**. La división conserva aproximadamente el porcentaje de cada clase."
    )
    conteos = pd.DataFrame({
        "Dataset completo": datos["aprobo"].value_counts().reindex([0, 1], fill_value=0),
        "Entrenamiento (80%)": y_train.value_counts().reindex([0, 1], fill_value=0),
        "Prueba (20%)": y_test.value_counts().reindex([0, 1], fill_value=0),
    })
    conteos.index = ["Desaprobó (0)", "Aprobó (1)"]
    st.dataframe(conteos, use_container_width=True)
    st.bar_chart(conteos[["Entrenamiento (80%)", "Prueba (20%)"]])
    with st.expander("Ver los primeros registros del CSV"):
        st.dataframe(datos.head(15), use_container_width=True)
    st.info(
        f"Se usó train_test_split(test_size=0.20, random_state={SEMILLA}, stratify=y). "
        "Los 120 registros de prueba quedan reservados para evaluar los modelos."
    )

with pestana_preparacion:
    st.subheader("Preparación de variables con ColumnTransformer")
    st.markdown("**Variables numéricas:** " + ", ".join(f"`{x}`" for x in COLUMNAS_NUMERICAS))
    st.write("Valores faltantes → mediana del entrenamiento; después → StandardScaler.")
    st.markdown("**Variables categóricas:** " + ", ".join(f"`{x}`" for x in COLUMNAS_CATEGORICAS))
    st.write("Valores faltantes → moda del entrenamiento; después → OneHotEncoder.")
    st.markdown("**Valores faltantes identificados en el dataset original**")
    nulos = datos.isna().sum().rename("Cantidad de nulos")
    st.dataframe(nulos.to_frame(), use_container_width=True)
    ejemplo = modelos["Regresión logística"].named_steps["preprocesamiento"]
    st.caption(
        f"Tras codificar, el modelo recibe {len(ejemplo.get_feature_names_out())} columnas. "
        "La cantidad puede variar si se utiliza otro CSV."
    )
    with st.expander("Mostrar nombres de las variables procesadas"):
        st.code("\n".join(ejemplo.get_feature_names_out()), language="text")
    st.info(
        "Para evitar fuga de información, el Pipeline ajusta medianas, categorías y "
        "escalado solo al entrenar con X_train. En X_test se aplica transform(), no fit()."
    )

with pestana_modelos:
    st.subheader("Primeros dos clasificadores")
    izquierda, derecha = st.columns(2)
    with izquierda:
        st.markdown("#### Regresión logística")
        st.write("`LogisticRegression(max_iter=1000, random_state=42)`")
        st.success("Pipeline entrenado correctamente")
        st.caption("Predice la clase 0 o 1 a partir de los datos del estudiante.")
    with derecha:
        st.markdown("#### K vecinos más cercanos (KNN)")
        st.write("`KNeighborsClassifier(n_neighbors=5)`")
        st.success("Pipeline entrenado correctamente")
        st.caption("Clasifica cada estudiante considerando cinco vecinos cercanos.")
    st.markdown(
        '<div class="indicacion"><b>Integración con la parte 3.</b> '
        'Ambos modelos reciben la misma separación 80/20. El equipo puede añadir '
        'SVC y calcular accuracy, precision, recall, F1 y matrices de confusión '
        'sin modificar la lógica de la partición.</div>',
        unsafe_allow_html=True,
    )
    st.code(
        "from ml_caso_a import separar_datos, construir_preprocesador\n"
        "from sklearn.pipeline import Pipeline\n"
        "from sklearn.svm import SVC\n\n"
        "X_train, X_test, y_train, y_test = separar_datos(datos)\n"
        "svm = Pipeline([\n"
        "    ('preprocesamiento', construir_preprocesador()),\n"
        "    ('modelo', SVC())\n"
        "])\n"
        "svm.fit(X_train, y_train)",
        language="python",
    )
    st.caption("Los parámetros iniciales no se han ajustado usando el conjunto de prueba.")

with pestana_prediccion:
    st.subheader("Probar los modelos con un estudiante de ejemplo")
    st.caption("Esta sección permite mostrar los modelos durante la sustentación; no es una decisión académica real.")
    c1, c2 = st.columns(2)
    with c1:
        horas = st.number_input("Horas de estudio por semana", min_value=0.0, max_value=80.0, value=9.0, step=0.5)
        asistencia = st.number_input("Asistencia (%)", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
        promedio = st.number_input("Promedio anterior (0 a 20)", min_value=0.0, max_value=20.0, value=12.0, step=0.5)
    with c2:
        tareas = st.number_input("Tareas entregadas", min_value=0, max_value=100, value=5, step=1)
        opciones_participacion = sorted(datos["participacion"].dropna().astype(str).unique())
        opciones_internet = sorted(datos["acceso_internet"].dropna().astype(str).unique())
        participacion = st.selectbox("Participación", opciones_participacion)
        internet = st.selectbox("Acceso a internet", opciones_internet)

    if st.button("Calcular predicción", type="primary"):
        ejemplo = pd.DataFrame([{
            "horas_estudio": horas,
            "asistencia": asistencia,
            "promedio_anterior": promedio,
            "tareas_entregadas": tareas,
            "participacion": participacion,
            "acceso_internet": internet,
        }])
        for nombre, modelo in modelos.items():
            clase = int(modelo.predict(ejemplo)[0])
            indice_riesgo = list(modelo.classes_).index(0)
            probabilidad = float(modelo.predict_proba(ejemplo)[0][indice_riesgo])
            estado = "Riesgo de desaprobar (0)" if clase == 0 else "Predicción: aprueba (1)"
            st.write(f"**{nombre}:** {estado} · Probabilidad estimada de clase 0: {probabilidad:.1%}")
        st.warning("Resultado demostrativo con datos sintéticos. No se debe usar para decidir sobre un estudiante real.")

st.divider()
st.caption("Curso: Sistemas Inteligentes y Machine Learning · Sesión 08 · Caso A · Datos sintéticos del docente")
