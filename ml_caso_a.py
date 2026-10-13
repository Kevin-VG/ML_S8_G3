"""Parte 2 - Partición y primeros modelos del Caso A (Sesión 08).

Este módulo no depende de Streamlit: se puede reutilizar al integrar SVM
para asegurar la misma división 80/20 y el mismo preprocesamiento.
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

COLUMNAS_NUMERICAS = [
    "horas_estudio", "asistencia", "promedio_anterior", "tareas_entregadas"
]
COLUMNAS_CATEGORICAS = ["participacion", "acceso_internet"]
VARIABLE_OBJETIVO = "aprobo"
COLUMNAS_REQUERIDAS = COLUMNAS_NUMERICAS + COLUMNAS_CATEGORICAS + [VARIABLE_OBJETIVO]
SEMILLA = 42
PORCENTAJE_PRUEBA = 0.20


def cargar_datos(ruta: str | Path) -> pd.DataFrame:
    """Carga el CSV y verifica que tenga las columnas necesarias."""
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
        raise ValueError("Se requieren al menos dos filas por clase para estratificar.")


def separar_datos(datos: pd.DataFrame):
    """Reserva el 20% como test y mantiene la proporción de clases."""
    validar_datos(datos)
    X = datos[COLUMNAS_NUMERICAS + COLUMNAS_CATEGORICAS].copy()
    y = datos[VARIABLE_OBJETIVO].copy()
    return train_test_split(
        X, y,
        test_size=PORCENTAJE_PRUEBA,
        random_state=SEMILLA,
        stratify=y,
    )


def construir_preprocesador() -> ColumnTransformer:
    """Toda imputación y normalización se ajusta solo al entrenar el Pipeline."""
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
    """Dos Pipelines INDEPENDIENTES, listos para fit(X_train, y_train)."""
    return {
        "Regresión logística": Pipeline(steps=[
            ("preprocesamiento", construir_preprocesador()),
            ("modelo", LogisticRegression(max_iter=1000, random_state=SEMILLA)),
        ]),
        "KNN": Pipeline(steps=[
            ("preprocesamiento", construir_preprocesador()),
            ("modelo", KNeighborsClassifier(n_neighbors=5)),
        ]),
    }


def entrenar_modelos(datos: pd.DataFrame):
    """Entrena sin tocar X_test: se devuelve test para evaluación final posterior."""
    X_train, X_test, y_train, y_test = separar_datos(datos)
    modelos = construir_modelos()
    for pipeline in modelos.values():
        pipeline.fit(X_train, y_train)
    return X_train, X_test, y_train, y_test, modelos
