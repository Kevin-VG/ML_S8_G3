# Parte 2 · Sesión 08 · Caso A

Proyecto Python con Streamlit para la **partición 80/20 y los primeros modelos** de identificación temprana de riesgo académico.

## Contenido
- `app.py`: interfaz Streamlit para presentar el trabajo y probar casos.
- `ml_caso_a.py`: funciones reutilizables para partición, preprocesamiento y entrenamiento.
- `estudiantes_sinteticos_s08.csv`: datos sintéticos entregados para la práctica.
- `requirements.txt`: dependencias.
- `.streamlit/config.toml`: tema visual oscuro / azul.

## Instalación y ejecución (Windows)

Desde PowerShell en la carpeta de este proyecto:

```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

También puede utilizarse `python -m pip` y `python -m streamlit` en sistemas donde `python` apunta a Python 3.

Streamlit mostrará una dirección local en la terminal (normalmente `http://localhost:8501`). No es necesario utilizar Google Colab.

## Método implementado
1. Variable objetivo: `aprobo` (0 = desaprobó, 1 = aprobó).
2. Seis variables predictoras.
3. `train_test_split(test_size=0.20, random_state=42, stratify=y)`.
4. `ColumnTransformer` para separar las cuatro numéricas de las dos categóricas.
5. `SimpleImputer(strategy='median')` + `StandardScaler` para numéricas.
6. `SimpleImputer(strategy='most_frequent')` + `OneHotEncoder(handle_unknown='ignore')` para categóricas.
7. `Pipeline` independiente para `LogisticRegression(max_iter=1000, random_state=42)` y `KNeighborsClassifier(n_neighbors=5)`.
8. El preprocesamiento se ajusta solo con el entrenamiento. El conjunto de prueba no interviene en la imputación ni el escalado.

### Resultado de la partición con el CSV incluido

| Clase | Entrenamiento | Prueba | Total |
|---|---:|---:|---:|
| 0 (desaprobó) | 155 | 39 | 194 |
| 1 (aprobó) | 325 | 81 | 406 |
| **Total** | **480** | **120** | **600** |

## Integración con la parte 3 (SVM + métricas)

`ml_caso_a.py` no depende de Streamlit, de modo que el compañero puede añadir un tercer modelo sin cambiar la lógica base:

```python
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from ml_caso_a import cargar_datos, separar_datos, construir_preprocesador

datos = cargar_datos('estudiantes_sinteticos_s08.csv')
X_train, X_test, y_train, y_test = separar_datos(datos)

svm = Pipeline([
    ('preprocesamiento', construir_preprocesador()),
    ('modelo', SVC()),
])
svm.fit(X_train, y_train)
```


