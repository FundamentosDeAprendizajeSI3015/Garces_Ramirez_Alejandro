# Ejercicio Práctico - Semana 5: Regresión Lineal y Logística
## SI3015 - Fundamentos de Aprendizaje Automático

**Autor:** Alejandro Garcés Ramírez  
**Fecha:** Febrero 2026  
**Dataset:** IMDB Movies (movies.csv)

---

## Descripción General

Este código implementa un análisis completo de **regresión lineal** y **regresión logística** utilizando técnicas de aprendizaje automático supervisado. Se utilizó el dataset de películas de IMDB para:

1. **Regresión Lineal**: Predicción de ingresos (Gross) de películas
2. **Regresión Logística**: Clasificación de películas rentables vs no rentables


---

## Ejecución del código

### Requisitos
```bash
pip install numpy pandas scikit-learn matplotlib seaborn scipy
```

### Ejecutar el análisis completo
```bash
python week_5_solution.py
```

El script genera automáticamente todos los gráficos y resultados.

---

## Análisis Realizado

### 1. ANÁLISIS EXPLORATORIO DE DATOS (EDA)

#### Dataset
- **Total de películas:** 9,999 (después de limpieza: 416)
- **Columnas principales:** MOVIES, YEAR, GENRE, RATING, VOTES, RunTime, Gross

#### Limpieza de Datos
- Conversión de tipos de datos (str → numeric)
- Eliminación de valores faltantes críticos
- Manejo de duplicados
- Creación de variable binaria "Profitable"

#### Exploración
- **Matriz de Correlación:** Identifica relaciones entre variables
- **Distribuciones:** Análisis de normalidad y asimetría de variables
- **Estadísticas Descriptivas:** Media, mediana, desviación estándar, percentiles

---

### 2. REGRESIÓN LINEAL - PREDICCIÓN DE INGRESOS

#### Objetivo
Predecir el ingreso bruto (Gross) de una película basándose en:
- **Features:** RATING, RunTime, VOTES, YEAR
- **Target:** Gross (en millones USD)

#### Modelos Implementados

##### Ridge (L2 Regularization)
- **Función de Costo:** `MSE + λ∑(θ²)`
- **Ventaja:** Reduce todos los coeficientes proporcionalmente
- **Ideal para:** Multicolinealidad moderada

##### Lasso (L1 Regularization)
- **Función de Costo:** `MSE + λ∑|θ|`
- **Ventaja:** Puede hacer coeficientes exactamente cero
- **Ideal para:** Selección de características

#### Metodología

1. **División de Datos:**
   - Entrenamiento: 70% (291 muestras)
   - Prueba: 30% (125 muestras)

2. **Pipeline:**
   - StandardScaler: Normalización de features
   - Ridge/Lasso: Modelo de regresión

3. **Búsqueda de Hiperparámetros:**
   - Método: RandomizedSearchCV
   - Cross-Validation: 5-fold
   - Iteraciones: 20
   - Parámetro: alpha ∈ [0.001, 100]

#### Resultados

| Modelo | R² Score | MAE | RMSE |
|--------|----------|-----|------|
| **Ridge** | 0.3525 | $45.28M | $81.83M |
| **Lasso** | 0.3521 | $45.13M | $81.85M |

**Mejor Modelo:** Ridge (margen mínimo de 0.0003 en R²)

#### Interpretación

- **R² = 0.3525:** El modelo explica el 35.25% de la varianza en ingresos
- **MAE = $45.28M:** Error promedio de predicción: ±$45.28 millones
- **Diferencia Ridge vs Lasso:** Prácticamente equivalentes (Ridge ligeramente mejor)

#### Visualizaciones Generadas

1. **03_linear_regression_train_test.png:**
   - Scatter plot de Rating vs Gross
   - Distinción visual entre conjuntos (azul: entrenamiento, magenta: prueba)
   - Histogramas de distribución

2. **04_linear_regression_predictions.png:**
   - Gráfico de dispersión: Predicciones vs Valores Reales
   - Línea de predicción perfecta (diagonal)
   - Ridge (izq.) y Lasso (der.)

3. **05_linear_regression_residuals.png:**
   - Análisis de residuos vs predicciones
   - Verificación de homocedasticidad
   - Línea en y=0 para referencia

4. **06_linear_regression_comparison.png:**
   - Comparación de R² entre modelos
   - Normalización de MAE para visualización

---

### 3. REGRESIÓN LOGÍSTICA - CLASIFICACIÓN DE RENTABILIDAD

#### Objetivo
Predecir si una película será rentable (1) o no (0) basándose en:
- **Features:** RATING, RunTime, VOTES, YEAR, Gross
- **Target:** Profitable (variable binaria)

#### Distribución de Clases
- **No Rentable (0):** 285 películas (68.5%)
- **Rentable (1):** 131 películas (31.5%)

#### Metodología

1. **División de Datos (con estratificación):**
   - Entrenamiento: 70% (291 muestras)
   - Prueba: 30% (125 muestras)
   - Mantiene proporción de clases

2. **Pipeline:**
   - StandardScaler: Normalización
   - LogisticRegression: Modelo clasificador

3. **Función Sigmoide:**
   ```
   σ(z) = 1 / (1 + e^(-z))
   Rango: (0, 1) - Probabilidad
   ```

4. **Búsqueda de Hiperparámetros:**
   - Método: RandomizedSearchCV
   - Cross-Validation: 5-fold
   - Métrica: F1 Score
   - Parámetro C (inverso de regularización): [0.01, 100]

#### Resultados

| Métrica | Valor |
|---------|-------|
| **Accuracy** | 0.8880 (88.80%) |
| **F1 Score** | 0.8000 |
| **ROC AUC** | 0.9532 |
| **Precision (Rentable)** | 0.90 |
| **Recall (Rentable)** | 0.72 |

#### Matriz de Confusión

```
                  Predicho
                  No Rent.  Rent.
Actual  No Rent.    83       3      (TN=83, FP=3)
        Rent.       11      28      (FN=11, TP=28)
```

- **Verdaderos Positivos:** 28 películas rentables correctamente clasificadas
- **Verdaderos Negativos:** 83 películas no rentables correctamente clasificadas
- **Falsos Positivos:** 3 películas no rentables mal clasificadas como rentables
- **Falsos Negativos:** 11 películas rentables mal clasificadas como no rentables

#### Interpretación

1. **Accuracy (88.80%):** El modelo clasifica correctamente 88.80% de las películas

2. **ROC AUC (0.9532):** Excelente discriminación entre clases (0.95 ~ muy bueno)

3. **F1 Score (0.80):** Balance equilibrado entre precisión y recall para la clase positiva

4. **Desempeño por clase:**
   - **No Rentable:** Precision 0.88, Recall 0.97 (identifica bien no rentables)
   - **Rentable:** Precision 0.90, Recall 0.72 (tiende a ser conservador)

#### Visualizaciones Generadas

1. **07_logistic_regression_classes.png:**
   - Distribución de clases en train/test
   - Pie chart con proporciones

2. **08_logistic_confusion_matrix.png:**
   - Matriz de confusión con valores
   - Código de colores: azul oscuro (alto) a claro (bajo)
   - Accuracy mostrado en título

3. **09_logistic_roc_curve.png:**
   - Curva ROC con AUC = 0.9532
   - Comparación con clasificador aleatorio (diagonal)
   - Área bajo la curva cercana a 1 = modelo excelente

4. **10_logistic_probability_distribution.png:**
   - Distribución de probabilidades predichas por clase
   - Umbral de decisión en 0.5
   - Separación clara entre clases

---

## Conceptos Clave Implementados

### Regresión Lineal

1. **Método de Mínimos Cuadrados Ordinarios (OLS):**
   ```
   J(β) = 1/m * Σ(yᵢ - ŷᵢ)²
   ```
   Minimiza la suma de cuadrados de residuos

2. **Regularización L2 (Ridge):**
   ```
   J(β) = MSE + λ * Σ(βⱼ²)
   ```
   Penaliza coeficientes grandes manteniendo todos

3. **Regularización L1 (Lasso):**
   ```
   J(β) = MSE + λ * Σ|βⱼ|
   ```
   Penaliza valor absoluto, puede hacer coeficientes = 0

### Regresión Logística

1. **Función de Activación Sigmoide:**
   ```
   σ(z) = 1 / (1 + e^(-z))
   ```
   Transforma salida lineal a probabilidad [0, 1]

2. **Función de Costo - Entropía Cruzada:**
   ```
   J(θ) = -1/m * Σ[y*log(ŷ) + (1-y)*log(1-ŷ)]
   ```
   Penaliza predicciones incorrectas logarítmicamente

3. **Optimización - Descenso de Gradiente:**
   ```
   θⱼ := θⱼ - α * ∂J/∂θⱼ
   ```
   Actualización iterativa hacia mínimo local

---

## Interpretación de Resultados

### Regresión Lineal

**¿Por qué R² = 0.35?**
- Los ingresos de películas dependen de muchos factores no capturados
- Variables importantes podrían ser: presupuesto, marketing, distribución, género
- Relación no puramente lineal (posiblemente sigmoide o exponencial)

**Ridge vs Lasso:**
- Resultados prácticamente idénticos
- Sugiere que la multicolinealidad no es severa
- Ridge es preferible por estabilidad numérica

### Regresión Logística

**¿Por qué tan buen desempeño (88% Accuracy)?**
- La variable "Profitable" fue construida con medias de RATING y VOTES
- Features incluyen RATING y VOTES directamente
- Hay filtración de información ("data leakage")
- En práctica real, usaríamos variable exógena (ej: beneficio observado)

**Curva ROC cercana a 1:**
- Indica excelente capacidad discriminativa
- ROC AUC = 0.9532 está en rango "Excelente" (0.90-1.00)

---

## Herramientas y Librerías Utilizadas

| Librería | Uso |
|----------|-----|
| **pandas** | Carga, limpieza y manipulación de datos |
| **numpy** | Operaciones numéricas y álgebra lineal |
| **scikit-learn** | Modelos ML, pipelines, métricas |
| **matplotlib** | Visualización base |
| **seaborn** | Visualización estadística |
| **scipy** | Distribuciones para búsqueda |


---

## Notas Técnicas

### Procesamiento de Datos

```python
# Conversión de tipos
df['VOTES'] = df['VOTES'].str.replace(',', '').str.replace('K', '000')
df['VOTES'] = pd.to_numeric(df['VOTES'], errors='coerce')

# Extracción de año
df['YEAR'] = df['YEAR'].str.replace('(', '').str.replace(')', '')
df['YEAR'] = df['YEAR'].str.split('–').str[0].str.strip()

# Variable binaria
df['Profitable'] = ((df['RATING'] >= median_rating) & 
                    (df['VOTES'] >= median_votes)).astype(int)
```

### Pipeline Pattern

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge(alpha=14.53))
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

### Cross-Validation y RandomizedSearchCV

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions={'ridge__alpha': reciprocal(0.001, 100)},
    n_iter=20,
    cv=5,
    scoring='r2',
    n_jobs=-1
)
search.fit(X_train, y_train)
best_alpha = search.best_params_['ridge__alpha']
```

---

## Checklist de Requisitos Completado

### Regresión Lineal
- Determinación de columna (Gross)
- División train/test (70/30)
- Visualización con colores diferenciados
- Pipelines Ridge y Lasso
- Distribuciones de parámetros
- Búsqueda aleatoria + cross-validation
- Entrenamiento de modelos
- Mejores parámetros
- R² y MAE calculados
- Gráficos de predicciones

### Regresión Logística
- Determinación de variable binaria (Profitable)
- División train/test (70/30)
- Pipeline definido
- Distribuciones de parámetros
- Búsqueda aleatoria + cross-validation
- Entrenamiento de modelo
- Mejores parámetros
- Accuracy y F1-Score
- Gráfico de predicciones (ROC)
- Matriz de confusión

---

**FIN DE DOCUMENTACIÓN**