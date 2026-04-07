# Ejercicio Práctico - Semana 2: Análisis del Dataset Iris
## SI3015 - Fundamentos de Aprendizaje Automático

**Autor:** Alejandro Garcés Ramírez
**Fecha:** Febrero 2026
**Dataset:** Iris Dataset (scikit-learn)

---

## Descripción General

Este código implementa el ciclo completo de aprendizaje automático supervisado sobre el clásico dataset Iris. El objetivo es introducir los conceptos fundamentales del flujo de trabajo ML, desde la carga de datos hasta la evaluación y persistencia del modelo.

---

## Ejecución del Código

### Requisitos
```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib
```

### Ejecutar el análisis
```bash
python Garcés_Ramírez_Alejandro_iris_analisis.py
```

---

## Análisis Realizado

### 1. Carga de Datos y EDA

- **Dataset:** 150 muestras de flores Iris con 4 características (sepal length, sepal width, petal length, petal width)
- **Clases:** 3 especies balanceadas (Setosa, Versicolor, Virginica — 50 muestras c/u)
- **Exploración:** Estadísticas descriptivas, distribución de clases, correlaciones

### 2. Ingeniería de Características

Variables derivadas creadas a partir de las originales:
- `sepal_ratio` = sepal length / sepal width
- `petal_ratio` = petal length / petal width
- `sepal_area` = sepal length × sepal width
- `petal_area` = petal length × petal width

### 3. Preprocesamiento

- **Escalado:** StandardScaler aplicado sobre el conjunto de entrenamiento (sin filtración al test)
- **División:** 70% entrenamiento / 30% prueba con estratificación por clase

### 4. Modelos Entrenados

| Modelo | Descripción |
|--------|-------------|
| Logistic Regression | Clasificador lineal probabilístico |
| K-Nearest Neighbors | Basado en distancia a vecinos |
| Decision Tree | Reglas de decisión jerárquicas |
| Random Forest | Ensamble de árboles (bagging) |
| SVM | Margen máximo de separación |

### 5. Búsqueda de Hiperparámetros

- **Método:** GridSearchCV
- **Validación cruzada:** 5-fold estratificada
- **Modelos tuneados:** SVM (`C`, `gamma`, `kernel`) y Random Forest (`n_estimators`, `max_depth`)

### 6. Evaluación

- **Métricas:** Accuracy, Precision, Recall, F1-Score, ROC-AUC (One-vs-Rest)
- **Validación cruzada:** K-Fold para estimar generalización
- **Visualizaciones:** Matrices de confusión, curvas ROC, comparación de modelos

### 7. Persistencia del Modelo

El mejor modelo se guarda en formato `.joblib` para reutilización posterior.

---

## Conceptos Clave Implementados

### Representación X, y
```python
# X: matriz de características (features)
X = iris.data           # shape (150, 4)

# y: vector de etiquetas (target)
y = iris.target         # shape (150,) → {0, 1, 2}
```

### Pipeline de Preprocesamiento
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SVC())
])
```

### Prevención de Filtración de Datos
```python
# CORRECTO: el scaler se ajusta solo en entrenamiento
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)  # aplica misma transformación

# INCORRECTO (data leakage):
# X_scaled = scaler.fit_transform(X)  # ← nunca antes del split
```

---

## Herramientas y Librerías

| Librería | Uso |
|----------|-----|
| **pandas** | Carga y manipulación de datos |
| **numpy** | Operaciones numéricas |
| **scikit-learn** | Modelos ML, métricas, pipelines |
| **matplotlib** | Visualización base |
| **seaborn** | Visualización estadística |
| **joblib** | Serialización del modelo |

---

## Notas Técnicas

- El dataset Iris no presenta valores faltantes → no requiere imputación
- Las clases están perfectamente balanceadas → accuracy es una métrica confiable
- Setosa es linealmente separable de Versicolor y Virginica → los modelos lineales también funcionan bien
- Versicolor y Virginica tienen cierto solapamiento → se espera mayor confusión entre ellas

---

**FIN DE DOCUMENTACIÓN**
