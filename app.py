"""Interfaz interactiva Streamlit — Sesión 08 (Caso A y Caso B)."""
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay

import ml_caso_a
import ml_caso_b

st.set_page_config(
    page_title="Machine Learning · Sesión 08",
    page_icon="⚡",
    layout="wide",
)

st.markdown("""
<style>
.block-container {padding-top: 1.8rem; max-width: 1200px;}
.subtitulo {color: #a8c7e5; margin-top: -8px; margin-bottom: 20px;}
.metric-box {
    background-color: #1a2332;
    padding: 15px;
    border-radius: 8px;
    border: 1px solid #2d3f59;
    margin-bottom: 12px;
}
.teoria-box {
    background-color: #172433;
    padding: 18px;
    border-left: 4px solid #3b82f6;
    border-radius: 6px;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# RENDERIZADO DEL CASO A (CLASIFICACIÓN)
# ==============================================================================
def render_caso_a():
    st.title("🎓 Identificación temprana de riesgo académico")
    st.markdown(
        '<p class="subtitulo">Sesión 08 · Caso A · '
        'Entrenamiento y evaluación de clasificadores</p>',
        unsafe_allow_html=True,
    )

    st.sidebar.header("Datos del Caso A")
    st.sidebar.caption(
        "Por defecto se utiliza el archivo sintético proporcionado para la práctica."
    )
    archivo_cargado = st.sidebar.file_uploader(
        "Usar otro CSV para Caso A (opcional)", type="csv", key="uploader_a"
    )

    try:
        if archivo_cargado is not None:
            datos = pd.read_csv(archivo_cargado)
            ml_caso_a.validar_datos(datos)
        else:
            ruta_csv = Path(__file__).resolve().parent / "estudiantes_sinteticos_s08.csv"
            datos = ml_caso_a.cargar_datos(ruta_csv)

        X_train, X_test, y_train, y_test, modelos = ml_caso_a.entrenar_modelos(datos)
        tabla_metricas, matrices = ml_caso_a.evaluar_modelos(modelos, X_test, y_test)
    except Exception as error:
        st.error(f"No se pudo iniciar el Caso A: {error}")
        return

    st.sidebar.success("Datos cargados y tres modelos entrenados")
    st.sidebar.caption(
        f"Semilla: {ml_caso_a.SEMILLA} · Train/test: 80/20 · Stratify: aprobo"
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
            f"Se usó train_test_split(test_size=0.20, random_state={ml_caso_a.SEMILLA}, "
            f"stratify=y). Los {len(X_test)} registros de prueba "
            "se reservan para la evaluación final."
        )

    with pestana_preparacion:
        st.subheader("Preparación de variables con ColumnTransformer")
        st.markdown(
            "**Variables numéricas:** "
            + ", ".join(f"`{col}`" for col in ml_caso_a.COLUMNAS_NUMERICAS)
        )
        st.write(
            "Valores faltantes → mediana del entrenamiento; después → StandardScaler."
        )
        st.markdown(
            "**Variables categóricas:** "
            + ", ".join(f"`{col}`" for col in ml_caso_a.COLUMNAS_CATEGORICAS)
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
            f"Tras codificar, el modelo recibe {len(nombres)} columnas."
        )
        with st.expander("Mostrar nombres de las variables procesadas"):
            st.code("\n".join(nombres), language="text")

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

    with pestana_prediccion:
        st.subheader("Probar los modelos con un estudiante de ejemplo")
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

        if st.button("Calcular predicción", type="primary", key="btn_pred_a"):
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
                    mensaje += f" · Probabilidad clase 0: {probabilidad:.1%}"
                st.write(mensaje)

    with pestana_evaluacion:
        st.subheader("Evaluación en el conjunto de prueba")
        st.dataframe(
            tabla_metricas.set_index("Modelo").round(4),
            use_container_width=True,
        )
        st.download_button(
            "Descargar métricas CSV",
            data=tabla_metricas.to_csv(index=False).encode("utf-8-sig"),
            file_name="metricas_caso_a.csv",
            mime="text/csv",
            key="dl_metricas_a",
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
            key="dl_matrices_a",
        )


# ==============================================================================
# RENDERIZADO DEL CASO B (REGRESIÓN)
# ==============================================================================
def render_caso_b():
    st.title("⚡ Predicción de Consumo Eléctrico Residencial")
    st.markdown(
        '<p class="subtitulo">Sesión 08 · Caso B · '
        'Estimación de consumo mensual (kWh) mediante Regresión Lineal</p>',
        unsafe_allow_html=True,
    )

    st.sidebar.header("Datos del Caso B")
    st.sidebar.caption(
        "Por defecto se utiliza el dataset cargado 'energia_sintetica_s08.csv'."
    )
    archivo_cargado = st.sidebar.file_uploader(
        "Usar otro CSV para Caso B (opcional)", type="csv", key="uploader_b"
    )

    try:
        if archivo_cargado is not None:
            datos = pd.read_csv(archivo_cargado)
            ml_caso_b.validar_datos(datos)
        else:
            ruta_csv = Path(__file__).resolve().parent / "energia_sintetica_s08.csv"
            datos = ml_caso_b.cargar_datos(ruta_csv)

        X_train, X_test, y_train, y_test, pipeline = ml_caso_b.entrenar_modelo(datos)
        evaluacion = ml_caso_b.evaluar_modelo(pipeline, X_test, y_test)
        detalles = ml_caso_b.obtener_detalles_modelo(pipeline)
    except Exception as error:
        st.error(f"No se pudo iniciar el Caso B: {error}")
        return

    st.sidebar.success("Dataset cargado y Regresión Lineal entrenada")
    st.sidebar.caption(
        f"Semilla: {ml_caso_b.SEMILLA} · Train/test: 80/20 · Modelo: LinearRegression"
    )

    res1, res2, res3, res4 = st.columns(4)
    res1.metric("Viviendas registradas", f"{len(datos):,}")
    res2.metric("Entrenamiento (80%)", f"{len(X_train):,}")
    res3.metric("Prueba (20%)", f"{len(X_test):,}")
    res4.metric("R² en prueba", f"{evaluacion['r2']:.4f}")

    (
        pestana_eda, pestana_preparacion, pestana_modelo,
        pestana_simulador, pestana_evaluacion,
    ) = st.tabs([
        "01 · Datos y EDA",
        "02 · Partición y Preprocesamiento",
        "03 · Modelo y Ecuación",
        "04 · Simulador de Consumo",
        "05 · Evaluación y Análisis Teórico",
    ])

    # --------------------------------------------------------------------------
    # PESTAÑA 1: EDA
    # --------------------------------------------------------------------------
    with pestana_eda:
        st.subheader("Exploración y Análisis de Variables (EDA)")
        st.write(
            "El objetivo es estimar el **consumo mensual de electricidad en kWh** "
            "a partir de las características físicas de la vivienda y hábitos de uso de sus ocupantes."
        )

        st.markdown("#### Identificación de Variables")
        desc_variables = pd.DataFrame([
            {"Variable": "area_m2", "Rol": "Predictora (X1)", "Tipo": "Numérica continua", "Descripción": "Superficie total de la vivienda en metros cuadrados."},
            {"Variable": "ocupantes", "Rol": "Predictora (X2)", "Tipo": "Numérica discreta", "Descripción": "Cantidad de personas que habitan en la vivienda."},
            {"Variable": "electrodomesticos", "Rol": "Predictora (X3)", "Tipo": "Numérica discreta", "Descripción": "Cantidad de electrodomésticos y artefactos eléctricos."},
            {"Variable": "horas_uso", "Rol": "Predictora (X4)", "Tipo": "Numérica continua", "Descripción": "Horas promedio diarias de uso intensivo de aparatos."},
            {"Variable": "consumo_kwh", "Rol": "Objetivo (Y)", "Tipo": "Numérica continua", "Descripción": "Consumo total de energía eléctrica mensual en kWh."},
        ])
        st.dataframe(desc_variables.set_index("Variable"), use_container_width=True)

        col_izq, col_der = st.columns(2)
        with col_izq:
            st.markdown("#### Primeros registros del Dataset")
            st.dataframe(datos.head(10), use_container_width=True)

            nulos = datos.isna().sum().to_frame(name="Valores nulos")
            st.markdown("#### Valores faltantes detectados")
            st.dataframe(nulos, use_container_width=True)
            if nulos.loc["area_m2", "Valores nulos"] > 0:
                st.warning(
                    f"Se detectaron {nulos.loc['area_m2', 'Valores nulos']} valores nulos en `area_m2`. "
                    "Se utilizará imputación por mediana durante el preprocesamiento."
                )

        with col_der:
            st.markdown("#### Estadísticas descriptivas generales")
            st.dataframe(datos.describe().round(2), use_container_width=True)

            st.markdown("#### Matriz de correlación lineal con 'consumo_kwh'")
            correlaciones = datos.corr()[["consumo_kwh"]].round(4)
            st.dataframe(correlaciones, use_container_width=True)

        st.markdown("#### Gráfico de Correlación de Variables con el Consumo (kWh)")
        corr_series = datos.corr()["consumo_kwh"].drop("consumo_kwh")
        st.bar_chart(corr_series)
        st.caption(
            "Todas las variables presentan correlaciones positivas moderadas-altas con el consumo eléctrico, "
            "lo que respalda firmemente el uso de un modelo lineal."
        )

    # --------------------------------------------------------------------------
    # PESTAÑA 2: PARTICIÓN Y PREPROCESAMIENTO
    # --------------------------------------------------------------------------
    with pestana_preparacion:
        st.subheader("Partición 80/20 y Estrategia de Preprocesamiento")

        st.markdown("#### 1. Partición de Datos (Train / Test Split)")
        particion_df = pd.DataFrame({
            "Conjunto": ["Entrenamiento (80%)", "Prueba (20%)", "Total"],
            "Registros": [len(X_train), len(X_test), len(datos)],
            "Porcentaje": ["80.0%", "20.0%", "100.0%"],
        })
        st.dataframe(particion_df.set_index("Conjunto"), use_container_width=True)

        st.markdown("""
        <div class="teoria-box">
        <strong>💡 ¿Por qué NO se utiliza 'stratify' en este caso de regresión?</strong><br>
        En clasificación (como en el Caso A), el parámetro <code>stratify=y</code> divide los datos asegurando que cada clase
        discreta (0 y 1) conserve la misma proporción en train y test.<br>
        En problemas de <strong>regresión</strong>, la variable objetivo (<code>consumo_kwh</code>) es <strong>continua e infinita</strong>,
        por lo que no existen clases discretas para estratificar. En su lugar, se realiza un muestreo aleatorio uniforme
        representativo con una semilla fija (<code>random_state=42</code>).
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 2. Preprocesamiento con Pipeline y SimpleImputer")
        st.write(
            "Para no perder las 10 filas que presentan datos faltantes en `area_m2`, "
            "se integra un `SimpleImputer(strategy='median')` en el Pipeline:"
        )
        st.code("""
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

pipeline = Pipeline(steps=[
    ('imputacion', SimpleImputer(strategy='median')),
    ('regresion', LinearRegression())
])
pipeline.fit(X_train, y_train)
        """, language="python")

        st.info(
            "🛡️ **Prevención de Data Leakage (Fuga de Información):**\n"
            "El valor de la mediana se calcula exclusivamente sobre `X_train`. El conjunto `X_test` no interviene "
            "en dicho cálculo; únicamente recibe la transformación con la mediana aprendida."
        )

    # --------------------------------------------------------------------------
    # PESTAÑA 3: MODELO Y ECUACIÓN
    # --------------------------------------------------------------------------
    with pestana_modelo:
        st.subheader("Ajuste del Modelo: LinearRegression")
        st.write(
            "La Regresión Lineal Múltiple modela la relación entre las variables predictoras y "
            "el consumo mensual buscando los coeficientes óptimos mediante el método de Mínimos Cuadrados Ordinarios (OLS)."
        )

        coefs = detalles["coeficientes"]
        b0 = detalles["intercepto"]

        st.markdown("#### Ecuación Matemática Aprendida por el Modelo:")
        st.latex(
            rf"\widehat{{\text{{consumo\_kwh}}}} = {b0:.4f} + {coefs['area_m2']:.4f}\cdot\text{{area\_m2}} "
            rf"+ {coefs['ocupantes']:.4f}\cdot\text{{ocupantes}} + {coefs['electrodomesticos']:.4f}\cdot\text{{electrodomesticos}} "
            rf"+ {coefs['horas_uso']:.4f}\cdot\text{{horas\_uso}}"
        )

        st.markdown("#### Interpretación Física y Práctica de los Parámetros:")
        tabla_coefs = pd.DataFrame([
            {
                "Término": "Intercepto (b₀)",
                "Valor": f"{b0:.4f} kWh",
                "Significado Práctico": "Consumo base residual mensual de una vivienda (consumo pasivo en reposo, conexiones fijas y equipos en stand-by)."
            },
            {
                "Término": "Área (b₁)",
                "Valor": f"+{coefs['area_m2']:.4f} kWh / m²",
                "Significado Práctico": "Por cada metro cuadrado adicional de vivienda, el consumo mensual aumenta aproximadamente 0.65 kWh (mayor iluminación y climatización)."
            },
            {
                "Término": "Ocupantes (b₂)",
                "Valor": f"+{coefs['ocupantes']:.4f} kWh / persona",
                "Significado Práctico": "Cada habitante adicional incrementa el consumo en ~15.44 kWh al mes (duchas eléctricas, artefactos personales, recargas)."
            },
            {
                "Término": "Electrodomésticos (b₃)",
                "Valor": f"+{coefs['electrodomesticos']:.4f} kWh / aparato",
                "Significado Práctico": "Cada equipo eléctrico adicional aporta en promedio ~8.84 kWh mensuales al consumo total."
            },
            {
                "Término": "Horas de uso (b₄)",
                "Valor": f"+{coefs['horas_uso']:.4f} kWh / hora diaria",
                "Significado Práctico": "Cada hora adicional de uso diario intensivo de equipos representa ~11.51 kWh más en el recibo mensual."
            },
        ])
        st.dataframe(tabla_coefs.set_index("Término"), use_container_width=True)

    # --------------------------------------------------------------------------
    # PESTAÑA 4: SIMULADOR
    # --------------------------------------------------------------------------
    with pestana_simulador:
        st.subheader("Simulador Interactivo de Consumo Residencial")
        st.caption("Modifica las variables de una vivienda para estimar su consumo mensual en kWh en tiempo real.")

        c1, c2 = st.columns(2)
        with c1:
            area_in = st.slider("Superficie de la vivienda (área m²)", min_value=30.0, max_value=250.0, value=115.0, step=1.0)
            ocupantes_in = st.slider("Cantidad de ocupantes (personas)", min_value=1, max_value=10, value=4, step=1)
        with c2:
            elect_in = st.slider("Cantidad de electrodomésticos", min_value=1, max_value=25, value=10, step=1)
            horas_in = st.slider("Horas promedio diarias de uso", min_value=1.0, max_value=16.0, value=6.5, step=0.5)

        if st.button("⚡ Calcular Consumo Estimado", type="primary", key="btn_pred_b"):
            ejemplo = pd.DataFrame([{
                "area_m2": area_in,
                "ocupantes": ocupantes_in,
                "electrodomesticos": elect_in,
                "horas_uso": horas_in,
            }])
            consumo_predicho = float(pipeline.predict(ejemplo)[0])

            st.markdown("---")
            col_m1, col_m2 = st.columns([1, 2])
            with col_m1:
                st.metric("Consumo Mensual Estimado", f"{consumo_predicho:.2f} kWh")
                if consumo_predicho < 250:
                    st.success("Categoría: Consumo Bajo (Tarifa reducida)")
                elif consumo_predicho <= 400:
                    st.info("Categoría: Consumo Moderado")
                else:
                    st.warning("Categoría: Consumo Alto (Alerta de eficiencia energética)")

            with col_m2:
                st.markdown("##### Desglose del Consumo por Factor:")
                aporte_base = detalles["intercepto"]
                aporte_area = area_in * coefs["area_m2"]
                aporte_ocupantes = ocupantes_in * coefs["ocupantes"]
                aporte_elect = elect_in * coefs["electrodomesticos"]
                aporte_horas = horas_in * coefs["horas_uso"]

                desglose = pd.DataFrame({
                    "Factor": ["Base residual", "Área m²", "Ocupantes", "Electrodomésticos", "Horas uso"],
                    "Aporte (kWh)": [aporte_base, aporte_area, aporte_ocupantes, aporte_elect, aporte_horas],
                })
                st.dataframe(desglose.set_index("Factor").round(2), use_container_width=True)

    # --------------------------------------------------------------------------
    # PESTAÑA 5: EVALUACIÓN Y PREGUNTAS TEÓRICAS
    # --------------------------------------------------------------------------
    with pestana_evaluacion:
        st.subheader("Evaluación del Modelo en Conjunto de Prueba (20%)")
        st.write(
            f"El modelo fue evaluado rigurosamente sobre las **{len(y_test)} viviendas reservadas** "
            "para prueba, las cuales nunca fueron vistas durante el entrenamiento."
        )

        st.markdown("#### Tabla de Métricas de Desempeño")
        st.dataframe(evaluacion["tabla_metricas"].set_index("Métrica"), use_container_width=True)

        st.download_button(
            "📥 Descargar métricas en CSV",
            data=evaluacion["tabla_metricas"].to_csv(index=False).encode("utf-8-sig"),
            file_name="metricas_caso_b_regresion.csv",
            mime="text/csv",
            key="dl_metricas_b",
        )

        st.markdown("---")
        st.subheader("📖 Preguntas y Fundamentos Teóricos de la Consigna")

        st.markdown("""
        <div class="teoria-box">
        <h4>1. Interpretación de Unidades de MAE y RMSE</h4>
        <ul>
            <li><strong>MAE (Error Absoluto Medio) = """ + f"{evaluacion['mae']:.2f}" + """ kWh:</strong>
            Se expresa en las <strong>mismas unidades que la variable original (kWh)</strong>. Representa el error promedio directo en valor absoluto. En la práctica, significa que las estimaciones difieren típicamente en ±""" + f"{evaluacion['mae']:.1f}" + """ kWh del consumo real mensual.</li>
            <li><strong>MSE (Error Cuadrático Medio) = """ + f"{evaluacion['mse']:.2f}" + """ kWh²:</strong>
            Se expresa en <strong>kWh al cuadrado</strong>. Por estar en unidades cuadráticas no tiene una lectura intuitiva en el recibo de luz, pero matemáticamente penaliza las discrepancias grandes.</li>
            <li><strong>RMSE (Raíz del Error Cuadrático Medio) = """ + f"{evaluacion['rmse']:.2f}" + """ kWh:</strong>
            Al aplicar la raíz cuadrada vuelve a las <strong>unidades originales (kWh)</strong>. Es especialmente útil porque <strong>penaliza con mayor severidad los errores atípicos o fallos grandes</strong>. Dado que el RMSE (""" + f"{evaluacion['rmse']:.2f}" + """ kWh) es algo mayor que el MAE (""" + f"{evaluacion['mae']:.2f}" + """ kWh), nos indica que existen algunas viviendas aisladas con errores mayores a la media.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(r"""
        <div class="teoria-box">
        <h4>2. ¿Por qué la Regresión Logística NO sirve para una variable continua como el consumo eléctrico?</h4>
        <ol>
            <li><strong>Propósito fundamental del algoritmo:</strong> La regresión logística es un modelo de <strong>clasificación</strong> (variables cualitativas discretas: aprobado/desaprobado, spam/no-spam), NO de regresión numérica continua.</li>
            <li><strong>Salida matemáticamente acotada en (0, 1):</strong> La regresión logística aplica la función sigmoide:
            <br><center>$$\sigma(z) = \frac{1}{1 + e^{-z}}$$</center>
            Su salida representa una <strong>probabilidad</strong> confinada obligatoriamente entre 0 y 1. Por lo tanto, es matemáticamente imposible que la regresión logística prediga valores reales como 150 kWh, 340.5 kWh o 500 kWh.</li>
            <li><strong>Función de pérdida inadecuada:</strong> La regresión logística optimiza la entropía cruzada binaria (Log-Loss) sobre clases discretas, mientras que predecir el consumo de energía exige minimizar el error cuadrático medio (MSE) o absoluto (MAE) sobre una recta real infinita en ℝ⁺.</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Gráficos de Diagnóstico: Valor Real vs. Estimado y Residuos")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Subplot 1: Real vs Predicho
        y_test_arr = y_test.to_numpy()
        y_pred_arr = evaluacion["y_pred"]
        min_val = min(y_test_arr.min(), y_pred_arr.min()) - 10
        max_val = max(y_test_arr.max(), y_pred_arr.max()) + 10

        ax1.scatter(y_test_arr, y_pred_arr, color="#3b82f6", alpha=0.75, edgecolors="#1e3a8a", s=50, label="Viviendas (Prueba)")
        ax1.plot([min_val, max_val], [min_val, max_val], color="#ef4444", linestyle="--", linewidth=2, label="Ajuste ideal (y = x)")
        ax1.set_title("Valores Reales vs. Estimados (kWh)", fontsize=13, fontweight="bold")
        ax1.set_xlabel("Consumo Real (kWh)", fontsize=11)
        ax1.set_ylabel("Consumo Predicho (kWh)", fontsize=11)
        ax1.set_xlim(min_val, max_val)
        ax1.set_ylim(min_val, max_val)
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper left")

        # Subplot 2: Gráfico de residuos
        residuos = evaluacion["residuos"]
        ax2.scatter(y_pred_arr, residuos, color="#10b981", alpha=0.75, edgecolors="#065f46", s=50)
        ax2.axhline(0, color="#ef4444", linestyle="--", linewidth=2, label="Residuo cero")
        ax2.set_title("Gráfico de Residuos (y_real - y_pred)", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Consumo Predicho (kWh)", fontsize=11)
        ax2.set_ylabel("Residuo (kWh)", fontsize=11)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="upper right")

        fig.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        st.pyplot(fig)
        plt.close(fig)

        st.download_button(
            "📥 Descargar gráficos diagnósticos para el informe (PNG)",
            data=buf.getvalue(),
            file_name="diagnostico_regresion_caso_b.png",
            mime="image/png",
            key="dl_graficos_b",
        )


# ==============================================================================
# ENRUTAMIENTO PRINCIPAL CON SELECTOR EN BARRA LATERAL
# ==============================================================================
st.sidebar.title("🧭 Navegación de Casos")
caso_seleccionado = st.sidebar.radio(
    "Selecciona la práctica a visualizar:",
    [
        "Caso A · Riesgo académico (Clasificación)",
        "Caso B · Consumo eléctrico (Regresión)",
    ],
    index=1,
)

if caso_seleccionado.startswith("Caso A"):
    render_caso_a()
else:
    render_caso_b()

st.divider()
st.caption(
    "Curso: Sistemas Inteligentes y Machine Learning · "
    "Sesión 08 · Comparativa Caso A y Caso B"
)