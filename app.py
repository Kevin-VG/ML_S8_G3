"""Interfaz Streamlit del Caso A — Sesión 08."""
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay

from ml_caso_a import (
    COLUMNAS_CATEGORICAS,
    COLUMNAS_NUMERICAS,
    SEMILLA,
    cargar_datos,
    entrenar_modelos,
    evaluar_modelos,
    validar_datos,
)

st.set_page_config(
    page_title="Riesgo académico | Caso A",
    page_icon="🎓",
    layout="wide",
)

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1180px;}
.subtitulo {color: #a8c7e5; margin-top: -8px; margin-bottom: 22px;}
</style>
""", unsafe_allow_html=True)

st.title("🎓 Identificación temprana de riesgo académico")
st.markdown(
    '<p class="subtitulo">Sesión 08 · Caso A · '
    'Entrenamiento y evaluación de clasificadores</p>',
    unsafe_allow_html=True,
)

st.sidebar.header("Datos del proyecto")
st.sidebar.caption(
    "Por defecto se utiliza el archivo sintético proporcionado para la práctica."
)
archivo_cargado = st.sidebar.file_uploader(
    "Usar otro CSV con las mismas columnas (opcional)", type="csv"
)

try:
    if archivo_cargado is not None:
        datos = pd.read_csv(archivo_cargado)
        validar_datos(datos)
    else:
        ruta_csv = Path(__file__).resolve().parent / "estudiantes_sinteticos_s08.csv"
        datos = cargar_datos(ruta_csv)

    X_train, X_test, y_train, y_test, modelos = entrenar_modelos(datos)
    tabla_metricas, matrices = evaluar_modelos(modelos, X_test, y_test)
except (
    FileNotFoundError, ValueError, KeyError, TypeError, pd.errors.ParserError
) as error:
    st.error(f"No se pudo iniciar el proyecto: {error}")
    st.stop()

st.sidebar.success("Datos cargados y tres modelos entrenados")
st.sidebar.caption(
    f"Semilla: {SEMILLA} · Train/test: 80/20 · Stratify: aprobo"
)

res1, res2, res3, res4 = st.columns(4)
res1.metric("Estudiantes", f"{len(datos):,}")
res2.metric("Entrenamiento", f"{len(X_train):,}")
res3.metric("Prueba", f"{len(X_test):,}")
res4.metric("Modelos entrenados", str(len(modelos)))

(
    pestana_datos, pestana_preparacion, pestana_modelos,
    pestana_prediccion, pestana_evaluacion,
) = st.tabs([
    "01 · Datos y partición",
    "02 · Preprocesamiento",
    "03 · Modelos entrenados",
    "04 · Predicción de ejemplo",
    "05 · Evaluación",
])

with pestana_datos:
    st.subheader("División 80% entrenamiento / 20% prueba")
    st.write(
        "La variable a predecir es **aprobo**: **0 = desaprobó (riesgo)** "
        "y **1 = aprobó**. La división conserva aproximadamente "
        "el porcentaje de cada clase."
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
        f"Se usó train_test_split(test_size=0.20, random_state={SEMILLA}, "
        f"stratify=y). Los {len(X_test)} registros de prueba "
        "se reservan para la evaluación final."
    )

with pestana_preparacion:
    st.subheader("Preparación de variables con ColumnTransformer")
    st.markdown(
        "**Variables numéricas:** "
        + ", ".join(f"`{col}`" for col in COLUMNAS_NUMERICAS)
    )
    st.write(
        "Valores faltantes → mediana del entrenamiento; después → StandardScaler."
    )
    st.markdown(
        "**Variables categóricas:** "
        + ", ".join(f"`{col}`" for col in COLUMNAS_CATEGORICAS)
    )
    st.write(
        "Valores faltantes → moda del entrenamiento; después → OneHotEncoder."
    )
    st.markdown("**Valores faltantes identificados en el dataset original**")
    st.dataframe(
        datos.isna().sum().rename("Cantidad de nulos").to_frame(),
        use_container_width=True,
    )
    preprocesador = modelos["Regresión logística"].named_steps["preprocesamiento"]
    nombres = preprocesador.get_feature_names_out()
    st.caption(
        f"Tras codificar, el modelo recibe {len(nombres)} columnas. "
        "La cantidad puede variar si se utiliza otro CSV."
    )
    with st.expander("Mostrar nombres de las variables procesadas"):
        st.code("\n".join(nombres), language="text")
    st.info(
        "Cada Pipeline ajusta medianas, categorías y escalado solo con X_train. "
        "En X_test aplica las transformaciones aprendidas sin volver a ajustarlas."
    )

with pestana_modelos:
    st.subheader("Tres clasificadores")
    configuraciones = {
        "Regresión logística": (
            "LogisticRegression(\n"
            "    max_iter=1000,\n"
            "    random_state=42\n"
            ")"
        ),
        "KNN": (
            "KNeighborsClassifier(\n"
            "    n_neighbors=5\n"
            ")"
        ),
        "SVM": (
            "SVC(\n"
            '    kernel="rbf",\n'
            "    C=1.0,\n"
            '    gamma="scale"\n'
            ")"
        ),
    }
    for columna, (nombre, configuracion) in zip(
        st.columns(3), configuraciones.items()
    ):
        with columna:
            st.markdown(f"#### {nombre}")
            st.code(configuracion, language="python")
            st.success("Pipeline entrenado correctamente")
    st.info(
        "Los tres modelos utilizan la misma partición 80/20. "
        "Cada uno tiene su propio Pipeline de preprocesamiento."
    )
    st.caption(
        "Se utilizan parámetros iniciales; el conjunto de prueba "
        "no se emplea para ajustar hiperparámetros."
    )

with pestana_prediccion:
    st.subheader("Probar los modelos con un estudiante de ejemplo")
    st.caption(
        "Esta sección permite demostrar los modelos durante la sustentación."
    )
    c1, c2 = st.columns(2)
    with c1:
        horas = st.number_input(
            "Horas de estudio por semana",
            min_value=0.0, max_value=80.0, value=9.0, step=0.5,
        )
        asistencia = st.number_input(
            "Asistencia (%)",
            min_value=0.0, max_value=100.0, value=75.0, step=1.0,
        )
        promedio = st.number_input(
            "Promedio anterior (0 a 20)",
            min_value=0.0, max_value=20.0, value=12.0, step=0.5,
        )
    with c2:
        tareas = st.number_input(
            "Tareas entregadas",
            min_value=0, max_value=100, value=5, step=1,
        )
        participacion = st.selectbox(
            "Participación",
            sorted(datos["participacion"].dropna().astype(str).unique()),
        )
        internet = st.selectbox(
            "Acceso a internet",
            sorted(datos["acceso_internet"].dropna().astype(str).unique()),
        )

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
            estado = (
                "Riesgo de desaprobar (0)" if clase == 0
                else "Predicción: aprueba (1)"
            )
            mensaje = f"**{nombre}:** {estado}"
            if hasattr(modelo, "predict_proba"):
                indice_riesgo = list(modelo.classes_).index(0)
                probabilidad = float(
                    modelo.predict_proba(ejemplo)[0][indice_riesgo]
                )
                mensaje += (
                    f" · Probabilidad estimada de clase 0: {probabilidad:.1%}"
                )
            st.write(mensaje)
        st.caption(
            "SVM muestra la clase predicha; su configuración "
            "actual no calcula probabilidades."
        )
        st.warning(
            "Resultado demostrativo con datos sintéticos. "
            "No se debe usar para decidir sobre un estudiante real."
        )

with pestana_evaluacion:
    st.subheader("Evaluación en el conjunto de prueba")
    st.write(
        f"Los tres modelos se evalúan sobre los mismos **{len(y_test)} "
        f"estudiantes**: **{int((y_test == 0).sum())} que desaprobaron** "
        f"y **{int((y_test == 1).sum())} que aprobaron**."
    )
    st.info(
        "Precision, recall y F1 consideran como clase positiva "
        "el riesgo académico: aprobo = 0. Accuracy considera ambas clases."
    )
    st.dataframe(
        tabla_metricas.set_index("Modelo").round(4),
        use_container_width=True,
    )
    st.download_button(
        "Descargar métricas CSV",
        data=tabla_metricas.to_csv(index=False).encode("utf-8-sig"),
        file_name="metricas_caso_a.csv",
        mime="text/csv",
    )

    st.subheader("Matrices de confusión")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (nombre, matriz) in zip(axes, matrices.items()):
        ConfusionMatrixDisplay(
            confusion_matrix=matriz,
            display_labels=["Riesgo (0)", "Aprobó (1)"],
        ).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(nombre)
        ax.set_xlabel("Clase predicha")
        ax.set_ylabel("Clase real")
    fig.tight_layout()
    imagen = BytesIO()
    fig.savefig(imagen, format="png", dpi=300, bbox_inches="tight")
    st.pyplot(fig)
    plt.close(fig)

    st.download_button(
        "Descargar matrices para el informe",
        data=imagen.getvalue(),
        file_name="matrices_confusion_caso_a.png",
        mime="image/png",
    )
    st.caption(
        "Filas: clase real. Columnas: clase predicha. "
        "Orden de clases: riesgo (0), aprobación (1)."
    )
    st.write(
        "**Falso negativo de riesgo:** estudiante que desaprobó, "
        "pero que el modelo predijo como aprobado. Corresponde "
        "a la celda superior derecha de cada matriz."
    )

st.divider()
st.caption(
    "Curso: Sistemas Inteligentes y Machine Learning · "
    "Sesión 08 · Caso A · Datos sintéticos del docente"
)