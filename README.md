# Sesión 08 · Casos de Machine Learning (A y B)

Proyecto Python con Streamlit para la resolución y sustentación de los casos prácticos de la Sesión 08:
* **Caso A (Clasificación):** Identificación temprana de riesgo académico (`aprobo`: 0 o 1).
* **Caso B (Regresión):** Predicción de consumo eléctrico mensual residencial en kWh (`consumo_kwh`).

---

## Contenido del Proyecto

- `app.py`: Interfaz interactiva en Streamlit con selector dinámico en la barra lateral para alternar entre el **Caso A** y el **Caso B**.
- `ml_caso_a.py`: Funciones modulares para partición estratificada 80/20, preprocesamiento (`ColumnTransformer`) y entrenamiento de clasificadores (Regresión Logística, KNN, SVM).
- `ml_caso_b.py`: Módulo independiente para partición 80/20, imputación con mediana, ajuste de `LinearRegression` y cálculo de métricas de regresión (MAE, MSE, RMSE, R²).
- `estudiantes_sinteticos_s08.csv`: Dataset sintético de 600 estudiantes para el Caso A.
- `energia_sintetica_s08.csv`: Dataset sintético de 450 viviendas para el Caso B.
- `requirements.txt`: Dependencias del entorno (`streamlit`, `pandas`, `scikit-learn`, `matplotlib`).

---

## Instalación y Ejecución (Windows)

Desde PowerShell en la carpeta raíz del proyecto:

```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

Streamlit abrirá automáticamente la aplicación en tu navegador en `http://localhost:8501`.

---

## Resumen Técnico del Caso B (Regresión)

### 1. Variables del Modelo
* **Predictoras:**
  * `area_m2`: Superficie de la vivienda en m² (con imputación de mediana para valores faltantes).
  * `ocupantes`: Número de habitantes.
  * `electrodomesticos`: Cantidad de equipos eléctricos.
  * `horas_uso`: Horas promedio diarias de uso intensivo.
* **Objetivo:**
  * `consumo_kwh`: Consumo eléctrico mensual en kWh (variable cuantitativa continua).

### 2. Flujo y Partición
* **Partición 80/20:** 360 viviendas para entrenamiento y 90 para prueba (`random_state=42`).
* **Nota Teórica:** En regresión **no** se utiliza `stratify`, ya que la variable objetivo es continua y carece de clases discretas.
* **Preprocesamiento:** `SimpleImputer(strategy='median')` ajustado únicamente en `X_train` para evitar fuga de información (*data leakage*).
* **Algoritmo:** `LinearRegression()` mediante Mínimos Cuadrados Ordinarios (OLS).

### 3. Ecuación Estimada y Coeficientes
$$\widehat{\text{consumo\_kwh}} \approx 30.7925 + 0.6501 \cdot \text{area} + 15.4370 \cdot \text{ocupantes} + 8.8412 \cdot \text{electrodomesticos} + 11.5093 \cdot \text{horas\_uso}$$

### 4. Métricas Obtenidas en el Conjunto de Prueba
* **$R^2$**: **0.8480** (El modelo explica el 84.80% de la variabilidad del consumo).
* **MAE**: **24.85 kWh** (Error absoluto promedio en kWh).
* **MSE**: **998.48 kWh²** (Error cuadrático medio).
* **RMSE**: **31.60 kWh** (Raíz del error cuadrático medio, penaliza desvíos grandes).

---

## Preguntas Teóricas del Caso B

### ¿Por qué la Regresión Logística no sirve para predecir el consumo de energía?
1. **Naturaleza del problema:** La regresión logística es un algoritmo de **clasificación** (categorías discretas). El consumo eléctrico es una **variable continua** cuantitativa en el rango $[0, \infty)\text{ kWh}$.
2. **Función Sigmoide acotada en $(0, 1)$:** La función sigmoide $\sigma(z) = \frac{1}{1 + e^{-z}}$ solo produce probabilidades entre 0 y 1. Es matemáticamente incapaz de predecir valores como $240\text{ kWh}$ o $410.5\text{ kWh}$.

### Interpretación de Unidades de MAE y RMSE
* **MAE:** Se expresa en **kWh**. Mide el error promedio en valor absoluto de las estimaciones del modelo.
* **RMSE:** También se expresa en **kWh**. Al elevar las diferencias al cuadrado antes de promediar, penaliza con mayor severidad los errores atípicos o fallas grandes.
