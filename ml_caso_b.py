"""Partición, preprocesamiento, entrenamiento y evaluación del Caso B (Regresión)."""
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

COLUMNAS_PREDICTORAS = [
    "area_m2",
    "ocupantes",
    "electrodomesticos",
    "horas_uso",
]
VARIABLE_OBJETIVO = "consumo_kwh"
COLUMNAS_REQUERIDAS = COLUMNAS_PREDICTORAS + [VARIABLE_OBJETIVO]
SEMILLA = 42
PORCENTAJE_PRUEBA = 0.20


def validar_datos(datos: pd.DataFrame) -> None:
    """Valida que el DataFrame contenga las columnas necesarias y tipos válidos."""
    faltan = [col for col in COLUMNAS_REQUERIDAS if col not in datos.columns]
    if faltan:
        raise ValueError("Faltan columnas en el CSV: " + ", ".join(faltan))
    
    if datos[VARIABLE_OBJETIVO].isna().any():
        raise ValueError("La variable objetivo 'consumo_kwh' no puede contener valores nulos.")
    
    if not pd.api.types.is_numeric_dtype(datos[VARIABLE_OBJETIVO]):
        raise ValueError("La variable objetivo 'consumo_kwh' debe ser de tipo numérico continuo.")

    for col in COLUMNAS_PREDICTORAS:
        if not pd.api.types.is_numeric_dtype(datos[col]):
            raise ValueError(f"La columna predictora '{col}' debe ser de tipo numérico.")

    if len(datos) < 20:
        raise ValueError("El dataset debe contener al menos 20 registros para realizar la partición 80/20.")


def cargar_datos(ruta: str | Path) -> pd.DataFrame:
    """Carga el dataset desde un archivo CSV y ejecuta la validación."""
    datos = pd.read_csv(ruta)
    validar_datos(datos)
    return datos


def separar_datos(datos: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Separa el 80% de los datos para entrenamiento y el 20% para prueba.
    En problemas de regresión continua NO se utiliza 'stratify'.
    """
    validar_datos(datos)
    X = datos[COLUMNAS_PREDICTORAS].copy()
    y = datos[VARIABLE_OBJETIVO].copy()
    return train_test_split(
        X, y, test_size=PORCENTAJE_PRUEBA, random_state=SEMILLA
    )


def construir_pipeline() -> Pipeline:
    """
    Construye un Pipeline con SimpleImputer (mediana) y LinearRegression.
    El imputador aprende la mediana de X_train y la aplica en X_test sin fuga de datos.
    """
    return Pipeline(steps=[
        ("imputacion", SimpleImputer(strategy="median")),
        ("regresion", LinearRegression()),
    ])


def entrenar_modelo(datos: pd.DataFrame):
    """
    Entrena el modelo de Regresión Lineal usando únicamente el conjunto de entrenamiento (80%).
    """
    X_train, X_test, y_train, y_test = separar_datos(datos)
    pipeline = construir_pipeline()
    pipeline.fit(X_train, y_train)
    return X_train, X_test, y_train, y_test, pipeline


def evaluar_modelo(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Evalúa el modelo en el conjunto de prueba (20%) calculando MAE, MSE, RMSE y R².
    """
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, y_pred)
    residuos = y_test.to_numpy() - y_pred

    tabla_metricas = pd.DataFrame([{
        "Métrica": "MAE (Error Absoluto Medio)",
        "Valor": round(mae, 4),
        "Unidad": "kWh",
        "Interpretación": "Desviación promedio absoluta entre lo predicho y lo real."
    }, {
        "Métrica": "MSE (Error Cuadrático Medio)",
        "Valor": round(mse, 4),
        "Unidad": "kWh²",
        "Interpretación": "Promedio de los errores al cuadrado (penaliza discrepancias grandes)."
    }, {
        "Métrica": "RMSE (Raíz Error Cuadrático Medio)",
        "Valor": round(rmse, 4),
        "Unidad": "kWh",
        "Interpretación": "Magnitud típica de los errores en las mismas unidades que la variable."
    }, {
        "Métrica": "R² (Coeficiente de Determinación)",
        "Valor": round(r2, 4),
        "Unidad": "Adimensional (0 a 1)",
        "Interpretación": f"El modelo explica el {r2 * 100:.2f}% de la variabilidad del consumo eléctrico."
    }])

    return {
        "tabla_metricas": tabla_metricas,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "y_pred": y_pred,
        "residuos": residuos,
    }


def obtener_detalles_modelo(pipeline: Pipeline) -> Dict[str, Any]:
    """Extrae el intercepto y los coeficientes asociados a cada variable."""
    modelo_reg = pipeline.named_steps["regresion"]
    coeficientes = dict(zip(COLUMNAS_PREDICTORAS, modelo_reg.coef_))
    intercepto = float(modelo_reg.intercept_)
    return {
        "intercepto": intercepto,
        "coeficientes": coeficientes,
    }
