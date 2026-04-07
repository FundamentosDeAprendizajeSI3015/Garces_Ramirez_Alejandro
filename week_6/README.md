# Ejercicio Práctico - Semana 6: Árboles de Decisión
## SI3015 - Fundamentos de Aprendizaje Automático

**Autor:** Alejandro Garcés Ramírez
**Fecha:** Marzo 2026
**Dataset:** Iris Dataset (scikit-learn)

---

## Descripción General

Este código implementa un análisis completo de **árboles de decisión** como método de clasificación supervisada. Se estudia el efecto del sobreajuste (overfitting) en árboles sin restricciones, la importancia del podado (pruning) y la búsqueda de hiperparámetros óptimos mediante GridSearchCV.

---

## Ejecución del Código

### Requisitos
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### Ejecutar el análisis
```bash
python solution.py
```

El script genera automáticamente todos los gráficos en la carpeta `outputs/`.

---

## Análisis Realizado

### 1. Dataset: Iris

| Característica | Valor |
|----------------|-------|
| **Muestras** | 150 (50 por clase) |
| **Características** | 4 (sepal length, sepal width, petal length, petal width) |
| **Clases** | 3 (Setosa, Versicolor, Virginica) |
| **Valores faltantes** | Ninguno |
| **Balance** | Clases perfectamente balanceadas |

---

### 2. Árbol de Decisión sin Podado

Un árbol sin restricciones de profundidad memoriza el conjunto de entrenamiento:

| Parámetro | Valor |
|-----------|-------|
| Profundidad máxima | Variable (sin límite) |
| N° de hojas | Variable |
| Accuracy (test) | ~95–97% |
| F1 Score (test) | ~95–97% |

**Problema:** Alta varianza → sensible a pequeños cambios en los datos de entrenamiento.

---

### 3. Búsqueda de Hiperparámetros (GridSearchCV)

```python
param_grid = {
    'max_depth': [2, 3, 4, 5, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}
# Cross-validation: StratifiedKFold(n_splits=5)
# Métrica: F1 Score ponderado
```

El espacio de búsqueda tiene **5 × 3 × 3 × 2 = 90 combinaciones** evaluadas con 5-fold CV.

---

### 4. Árbol de Decisión Optimizado

Con los mejores hiperparámetros encontrados por GridSearchCV:

| Parámetro | Valor |
|-----------|-------|
| Profundidad máxima | 3–4 (podado) |
| Accuracy (test) | ~97–98% |
| F1 Score (test) | ~97–98% |
| CV F1 (5-fold) | ~96–98% |

**Ventaja:** Árbol más simple, más interpretable y más generalizable.

---

### 5. Criterios de División

#### Impureza Gini
```
Gini(t) = 1 - Σᵢ pᵢ²
```
Mide la probabilidad de clasificar incorrectamente una muestra aleatoria. Un nodo puro tiene Gini = 0.

#### Entropía (Información Mutua)
```
H(t) = -Σᵢ pᵢ · log₂(pᵢ)
```
Mide el desorden informacional del nodo. Un nodo puro tiene H = 0.

#### Ganancia de Información
```
IG = H(padre) - Σ [|hijo| / |padre|] × H(hijo)
```
El algoritmo CART selecciona el split que maximiza la Ganancia de Información.

---

### 6. Importancia de Características

Las características más discriminativas del dataset Iris son:

1. **petal length (cm)** — separación principal entre Setosa y las demás
2. **petal width (cm)** — diferencia Versicolor de Virginica

Las características del sépalo (`sepal length`, `sepal width`) tienen menor poder discriminativo.

---

### 7. Interpretación de las Reglas del Árbol

```
petal length <= 2.45
├── class: setosa  ← perfectamente separable
└── petal width <= 1.75
    ├── class: versicolor
    └── class: virginica
```

Esta legibilidad es la **mayor ventaja** de los árboles sobre otros modelos: las reglas se pueden explicar en lenguaje natural.

---

## Visualizaciones Generadas

| Archivo | Descripción |
|---------|-------------|
| `01_eda_overview.png` | Distribución de clases + scatter pétalo |
| `02_feature_distributions.png` | Histogramas de características por clase |
| `03_correlation_matrix.png` | Matriz de correlación |
| `04_tree_full.png` | Árbol completo sin podado |
| `05_tree_pruned.png` | Árbol optimizado (podado) |
| `06_confusion_matrices.png` | Matrices de confusión (sin podado vs optimizado) |
| `07_feature_importance.png` | Importancia de características |
| `08_comparison.png` | Comparación de métricas |

---

## Conceptos Clave Implementados

### Sobreajuste vs Generalización

| | Árbol sin podado | Árbol optimizado |
|--|-----------------|-----------------|
| Complejidad | Alta | Baja |
| Accuracy (train) | ~100% | ~97% |
| Accuracy (test) | Variable | Estable |
| Interpretabilidad | Baja | Alta |

### Algoritmo CART (Classification and Regression Trees)

1. Para cada característica y cada umbral posible → calcular impureza del split
2. Seleccionar el split con mayor reducción de impureza (Ganancia de Información)
3. Dividir el nodo → crear dos nodos hijos
4. Repetir recursivamente hasta condición de parada (`max_depth`, `min_samples_leaf`, etc.)

---

## Herramientas y Librerías

| Librería | Uso |
|----------|-----|
| **scikit-learn** | `DecisionTreeClassifier`, `GridSearchCV`, `StratifiedKFold` |
| **sklearn.tree** | `plot_tree`, `export_text` |
| **pandas** | Manejo del DataFrame |
| **numpy** | Operaciones numéricas |
| **matplotlib** | Visualización base |
| **seaborn** | Heatmaps y estilo |

---

## Checklist de Requisitos

- División train/test con estratificación
- Árbol sin restricciones (baseline)
- Búsqueda de hiperparámetros (GridSearchCV + StratifiedKFold)
- Árbol con podado (best params)
- Reporte de clasificación completo
- Matrices de confusión
- Importancia de características
- Visualización del árbol (plot_tree)
- Reglas textuales (export_text)
- Cross-validation para estimación de generalización

---

**FIN DE DOCUMENTACIÓN**
