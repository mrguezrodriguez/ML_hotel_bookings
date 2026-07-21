# Predicción de cancelaciones de reservas hoteleras

> **Proyecto de Machine Learning · Clasificación binaria**
> Predicción de la probabilidad de cancelación de una reserva hotelera a partir de sus características, usando el dataset público *Hotel Booking Demand*.

---

## Índice

1. [Descripción del problema](#1-descripción-del-problema)
2. [Dataset utilizado](#2-dataset-utilizado)
3. [Solución adoptada](#3-solución-adoptada)
4. [Estructura del repositorio](#4-estructura-del-repositorio)
5. [Tecnologías utilizadas](#5-tecnologías-utilizadas)
6. [Instrucciones de reproducción](#6-instrucciones-de-reproducción)
7. [Principales resultados](#7-principales-resultados)
8. [Autores](#8-autores)

---

## 1. Descripción del problema

La rentabilidad de un hotel depende directamente de su nivel de ocupación real, es decir, del número de reservas que finalmente se materializan. Sin embargo, una proporción significativa de las reservas termina en cancelación, y cuando esta no se compensa con nuevas reservas en un periodo razonable, el hotel pierde ingresos y reduce su eficiencia operativa.

**Objetivo de negocio:** anticipar qué reservas tienen mayor probabilidad de cancelarse, para que los equipos de Revenue Management y Reservas puedan:

- aplicar estrategias de overbooking controlado,
- ajustar precios de forma dinámica,
- mejorar la comunicación con clientes de alto riesgo,
- adaptar las políticas de cancelación según el perfil de la reserva.

**Objetivo de Machine Learning:**

| | |
|---|---|
| Tipo de problema | Clasificación binaria |
| Variable target | `is_canceled` |
| Clase 0 | Reserva que llega a término |
| Clase 1 | Reserva cancelada |
| Métricas principales | ROC-AUC, F1-score, Recall, Precision |
| Umbral mínimo de utilidad | ROC-AUC ≥ 0.75 y Recall ≥ 0.70 (clase cancelación) |

El dataset está desbalanceado (~37% cancelaciones / 63% no cancelaciones), por lo que el Accuracy no se usa como métrica de referencia. Dado que el coste de no detectar una cancelación es mayor que el de una falsa alarma, se prioriza el **Recall** y el **F1-score** de la clase positiva sobre el Accuracy.

---

## 2. Dataset utilizado

**Fuente:** [Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) (Kaggle) — licencia Open Data, uso permitido para proyectos educativos.

El CSV original no se incluye en el repositorio (ver `.gitignore`). Para obtenerlo:

1. Configura tu token de Kaggle en `~/.kaggle/kaggle.json` (puedes generarlo desde [kaggle.com/settings](https://www.kaggle.com/settings)).
2. Ejecuta el script de descarga:
   ```bash
   python download_data.py
   ```
   Esto descarga el dataset y lo deja preparado en `src/data_sample/hotel_bookings.csv`.

---

## 3. Solución adoptada

Aproximación de Machine Learning supervisado con las siguientes fases, desarrolladas en `main.ipynb`:

| Fase | Contenido |
|---|---|
| **EDA dirigido al modelado** | Análisis del target, relación variable–target (correlación de Pearson para numéricas, V de Cramer para categóricas), variables temporales, multicolinealidad y outliers |
| **Feature Engineering** | Variables derivadas (`total_nights`, `is_family`, `total_guests`, `adr_per_person`), variables de fecha (`season`, codificación cíclica del mes) y discretización por cuartiles (`lead_time`, `adr`, `total_nights`) |
| **Preprocesado** | Eliminación de columnas con leakage/alta nulidad, imputación (moda/constante calculada solo sobre train), tratamiento de outliers (capado al percentil 99), One-Hot / Ordinal Encoding, escalado |
| **Feature Selection** | Importancia por Random Forest + filtro de multicolinealidad entre variables dummy |
| **Encapsulado en Pipeline** | Todo el preprocesado anterior reproducido mediante transformers custom de `scikit-learn` (`BaseEstimator`/`TransformerMixin`), con `fit` exclusivo sobre train para evitar fuga de información |
| **Evaluación de modelos candidatos** | Comparación de varios algoritmos de clasificación mediante validación cruzada (Random Forest, XGBoost, LightGBM, entre otros) |
| **Selección de modelo** | **XGBoost** es el modelo con mejor rendimiento global (ROC-AUC, F1, Precision y Recall), superando ligeramente a LightGBM en todas las métricas |
| **Optimización de hiperparámetros** | `RandomizedSearchCV` seguido de `GridSearchCV` para afinar el XGBoost seleccionado |
| **Persistencia** | Modelo final, pipeline completo y conjuntos preprocesados serializados con `joblib` (`src/models/`) para reutilizarlos sin reentrenar |
| **Evaluación final y puesta en marcha** | Evaluación sobre el conjunto de test (Classification Report, Matriz de Confusión, Curva ROC), análisis de Feature Importances, y validación del pipeline completo simulando la predicción sobre una reserva nueva |

Algunas decisiones metodológicas relevantes documentadas en el notebook:

- **Duplicados:** se detecta un ~27% de filas duplicadas en el dataset original. Al tratarse de un dataset anonimizado (sin identificador de reserva/cliente), no hay forma de distinguir duplicados reales de coincidencias legítimas, por lo que **se decide no eliminarlos**.
- **Outliers en variables discretas:** el método IQR no es adecuado para variables como `children`, `babies` o `total_of_special_requests`; se documentan pero no se transforman en esta fase.
- **Leakage:** `reservation_status` y `reservation_status_date` se descartan por coincidir con el target.

---

## 4. Estructura del repositorio

```
ML_hotel_bookings/
│
├── main.ipynb                 # Notebook principal: EDA, feature engineering,
│                               #   preprocesado, pipeline y modelado
├── download_data.py            # Script de descarga del dataset desde Kaggle
├── Presentacion.pdf              # Presentación del proyecto
├── requirements.txt             # Dependencias del proyecto
├── README.md
│
└── src/
    ├── data_sample/            # Dataset descargado (no versionado, ver .gitignore)
    │   └── hotel_bookings.csv
    ├── img/                     # Gráficas e imágenes generadas en el EDA
    ├── models/                  # Modelos y artefactos entrenados (joblib)
    │   ├── full_pipeline.pkl      # Pipeline completo de preprocesado
    │   ├── xgboost_final.pkl      # Modelo XGBoost optimizado (GridSearchCV)
    │   ├── X_train_prep.pkl / X_test_prep.pkl
    │   └── y_train.pkl / y_test.pkl
    ├── notebooks/                # Notebooks auxiliares / de referencia
    └── utils/                    # Utilidades y documentación de apoyo
```

---

## 5. Tecnologías utilizadas

| Categoría | Herramientas |
|---|---|
| Lenguaje | Python |
| Manipulación de datos | pandas ≥ 2.0, numpy ≥ 1.24 |
| Visualización | matplotlib ≥ 3.7, seaborn ≥ 0.12 |
| Machine Learning | scikit-learn ≥ 1.3, xgboost ≥ 2.0, lightgbm ≥ 4.0 |
| Serialización de modelos | joblib ≥ 1.3 |
| Datos | kagglehub ≥ 0.3.0 |
| Entorno | Jupyter Notebook, venv |

---

## 6. Instrucciones de reproducción

### Requisitos previos

- Python 3.x
- Cuenta de Kaggle con token API configurado

### Instalación

```bash
# Clonar el repositorio
git clone https://github.com/mrguezrodriguez/ML_hotel_bookings.git
cd ML_hotel_bookings

# Crear entorno virtual
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### Descarga de datos

```bash
python src/data_sample/download_data.py
```

### Ejecución

```bash
jupyter notebook main.ipynb
```

Ejecuta las celdas **en orden, de arriba a abajo**, sin saltar ninguna: el notebook construye el preprocesado de forma incremental sobre `X_train`/`X_test`, por lo que reejecutar celdas fuera de orden puede dejar los datos en un estado inconsistente.

> Si solo quieres reutilizar el modelo ya entrenado (sin reejecutar todo el notebook), puedes cargar directamente los artefactos persistidos en `src/models/`:
> ```python
> import joblib
> full_pipeline = joblib.load("src/models/full_pipeline.pkl")
> best_xgb = joblib.load("src/models/xgboost_final.pkl")
> ```

---

## 7. Principales resultados

Tras comparar varios algoritmos de clasificación mediante validación cruzada, **XGBoost** obtuvo el mejor rendimiento global (mayor ROC-AUC, F1-score, Precision y Recall), superando ligeramente a LightGBM en todas las métricas evaluadas.

**Optimización de hiperparámetros** (RandomizedSearchCV → GridSearchCV sobre XGBoost):

| Búsqueda | ROC-AUC |
|---|---|
| RandomizedSearchCV | 0.9496 |
| GridSearchCV | 0.9509 |

Configuración final (GridSearchCV): `colsample_bytree=0.75`, `learning_rate=0.1`, `max_depth=9`, `n_estimators=500`, `subsample=0.92`.

**Evaluación final:** el modelo alcanza un **ROC-AUC cercano a 0.953** sobre el conjunto de test, mostrando buena capacidad de generalización — ligeramente por encima incluso de los resultados de validación cruzada. Supera con margen el umbral mínimo definido en el objetivo del proyecto (ROC-AUC ≥ 0.75, Recall ≥ 0.70).

El *Classification Report* muestra un rendimiento equilibrado entre clases: la clase "no cancelada" obtiene precision/recall ligeramente superiores, mientras que la clase "cancelada" alcanza un **F1-score de 0.84**, sin sesgo relevante hacia ninguna de las dos. La matriz de confusión confirma un número reducido de falsos positivos y falsos negativos en proporción al total, y la curva ROC se aproxima claramente a la esquina superior izquierda, coherente con el ROC-AUC obtenido.

**Feature Importances:** `deposit_type` (tipo de depósito) es, con diferencia, la variable más influyente — en particular `deposit_type_Non Refund` (depósito no reembolsable). Le siguen, con contribución bastante menor, `deposit_type_No Deposit`, `required_car_parking_spaces`, `previous_cancellations` y `market_segment`.

**Puesta en marcha:** se valida el funcionamiento del pipeline completo simulando la predicción sobre una reserva nueva con variables originales — el pipeline se carga con `joblib` y transforma los datos automáticamente antes de pasarlos al modelo. Para la reserva de ejemplo simulada, el modelo predice que **no será cancelada**, con una probabilidad de cancelación del **0.12%** (99.88% de probabilidad de que se mantenga).

**Conclusión:** el modelo presenta potencial para integrarse en sistemas de gestión hotelera como herramienta de apoyo a la toma de decisiones sobre reservas con alto riesgo de cancelación, permitiendo anticipar estrategias de overbooking, precios dinámicos o comunicación preventiva con el cliente.

---

## 8. Autores

- **Melania Fondevilla** — [GitHub](https://github.com/mfondevilla)
- **William Walker** — [GitHub](https://github.com/willwalker7)
- **María Rodríguez** — [GitHub](https://github.com/mrguezrodriguez)
