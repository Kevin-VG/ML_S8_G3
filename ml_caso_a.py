"""Partición, preprocesamiento, entrenamiento y evaluación del Caso A."""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

COLUMNAS_NUMERICAS = [
    "horas_estudio", "asistencia",
    "promedio_anterior", "tareas_entregadas",
]
COLUMNAS_CATEGORICAS = ["participacion", "acceso_internet"]
VARIABLE_OBJETIVO = "aprobo"
COLUMNAS_REQUERIDAS = (
    COLUMNAS_NUMERICAS + COLUMNAS_CATEGORICAS + [VARIABLE_OBJETIVO]
)
SEMILLA = 42
PORCENTAJE_PRUEBA = 0.20


def cargar_datos(ruta: str | Path) -> pd.DataFrame:
    """Carga el CSV y verifica las columnas y la etiqueta."""
    datos = pd.read_csv(ruta)
    validar_datos(datos)
    return datos


def validar_datos(datos: pd.DataFrame) -> None:
    faltan = [col for col in COLUMNAS_REQUERIDAS if col not in datos.columns]
    if faltan:
        raise ValueError("Faltan columnas en el CSV: " + ", ".join(faltan))
    if datos[VARIABLE_OBJETIVO].isna().any():
        raise ValueError("La variable aprobo no puede tener valores faltantes.")
    if set(datos[VARIABLE_OBJETIVO].unique()) != {0, 1}:
        raise ValueError("La columna aprobo debe contener las clases 0 y 1.")
    if datos[VARIABLE_OBJETIVO].value_counts().min() < 2:
        raise ValueError("Se requieren al menos dos filas por clase.")


def separar_datos(datos: pd.DataFrame):
    """Reserva el 20% para prueba y conserva la proporción de clases."""
    validar_datos(datos)
    X = datos[COLUMNAS_NUMERICAS + COLUMNAS_CATEGORICAS].copy()
    y = datos[VARIABLE_OBJETIVO].copy()
    return train_test_split(
        X, y, test_size=PORCENTAJE_PRUEBA,
        random_state=SEMILLA, stratify=y,
    )


def construir_preprocesador() -> ColumnTransformer:
    """Los transformadores se ajustan dentro del Pipeline con train."""
    numericas = Pipeline(steps=[
        ("imputacion", SimpleImputer(strategy="median")),
        ("escalado", StandardScaler()),
    ])
    categoricas = Pipeline(steps=[
        ("imputacion", SimpleImputer(strategy="most_frequent")),
        ("codificacion", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("numericas", numericas, COLUMNAS_NUMERICAS),
        ("categoricas", categoricas, COLUMNAS_CATEGORICAS),
    ])


def construir_modelos() -> dict[str, Pipeline]:
    """Construye tres Pipelines con preprocesadores independientes."""
    clasificadores = {
        "Regresión logística": LogisticRegression(
            max_iter=1000, random_state=SEMILLA
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel="rbf", C=1.0, gamma="scale"),
    }
    return {
        nombre: Pipeline(steps=[
            ("preprocesamiento", construir_preprocesador()),
            ("modelo", clasificador),
        ])
        for nombre, clasificador in clasificadores.items()
    }


def entrenar_modelos(datos: pd.DataFrame):
    """Entrena los tres modelos usando solamente el conjunto train."""
    X_train, X_test, y_train, y_test = separar_datos(datos)
    modelos = construir_modelos()
    for pipeline in modelos.values():
        pipeline.fit(X_train, y_train)
    return X_train, X_test, y_train, y_test, modelos


def evaluar_modelos(modelos, X_test, y_test):
    """Calcula métricas en test; riesgo (0) es la clase positiva."""
    resultados, matrices = [], {}
    for nombre, pipeline in modelos.items():
        y_pred = pipeline.predict(X_test)
        resultados.append({
            "Modelo": nombre,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision riesgo (0)": precision_score(
                y_test, y_pred, pos_label=0, zero_division=0
            ),
            "Recall riesgo (0)": recall_score(
                y_test, y_pred, pos_label=0, zero_division=0
            ),
            "F1 riesgo (0)": f1_score(
                y_test, y_pred, pos_label=0, zero_division=0
            ),
        })
        matrices[nombre] = confusion_matrix(y_test, y_pred, labels=[0, 1])
    return pd.DataFrame(resultados), matrices