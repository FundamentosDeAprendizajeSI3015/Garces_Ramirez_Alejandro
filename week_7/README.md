# Ejercicio Práctico - Semana 7: Random Forest
## SI3015 - Fundamentos de Aprendizaje Automático

**Autor:** Alejandro Garcés Ramírez
**Fecha:** Marzo 2026
**Dataset:** Breast Cancer Wisconsin (scikit-learn)

---

## Descripción General

Este código implementa **Random Forest** como método de ensamble basado en árboles de decisión. Se compara su desempeño contra un árbol individual, se analiza el efecto del número de árboles, y se estudia la importancia de características en un problema de clasificación binaria médica.

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

### 1. Dataset: Breast Cancer Wisconsin

| Característica | Valor |
|----------------|-------|
| **Muestras** | 569 tumores mamarios |
| **Características** | 30 (radio, textura, perímetro, área, etc.) |
| **Clases** | 2 (Maligno=0, Benigno=1) |
| **Distribución** | ~37% Maligno, ~63% Benigno |
| **Valores faltantes** | Ninguno |

Las características describen propiedades del núcleo celular extraídas de imágenes digitalizadas de biopsias.

---

### 2. Árbol de Decisión (Baseline)

Modelo individual con `max_depth=5` como punto de referencia:

| Métrica | Valor |
|---------|-------|
| Accuracy | ~93–95% |
| F1 Score | ~93–95% |
| ROC-AUC | ~97% |

---

### 3. Búsqueda de Hiperparámetros (GridSearchCV)

```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'max_features': ['sqrt', 'log2'],
    'min_samples_leaf': [1, 2, 4]
}
# oob_score=True → error out-of-bag habilitado
# Cross-validation: StratifiedKFold(n_splits=5)
# Métrica: F1 Score ponderado
```

**72 combinaciones** evaluadas con validación cruzada estratificada de 5 pliegues.

---

### 4. Random Forest Optimizado

| Métrica | Árbol Individual | Random Forest | Diferencia |
|---------|-----------------|---------------|-----------|
| **Accuracy** | ~93–95% | ~96–98% | +2–3% |
| **F1 Score** | ~93–95% | ~96–98% | +2–3% |
| **ROC-AUC** | ~97% | ~99% | +1–2% |
| **OOB Score** | — | ~96–97% | — |

El OOB Score (Out-of-Bag) es una estimación gratuita del error de generalización usando los datos no usados en cada árbol bootstrap.

---

### 5. ¿Por qué Random Forest supera al árbol individual?

#### Bagging (Bootstrap Aggregating)

Cada árbol se entrena con una muestra aleatoria con reemplazo:
- ~63% de los datos se usan para entrenar (muestra bootstrap)
- ~37% quedan como "out-of-bag" (validación gratuita)

```
Árbol 1: muestras [1,3,3,7,9,2,...]  → predicción₁
Árbol 2: muestras [4,1,8,1,3,6,...]  → predicción₂
Árbol N: muestras [2,5,9,4,2,7,...]  → predicciónN
─────────────────────────────────────────────────
Votación: clase mayoritaria = predicción final
```

#### Aleatoriedad de Características

En cada split solo se evalúa un subconjunto de características (`max_features = sqrt(p)`):
- Decorrelaciona los árboles entre sí
- Reduce la varianza sin aumentar el sesgo significativamente

---

### 6. Importancia de Características

Random Forest proporciona una medida de importancia basada en la reducción media de impureza (Mean Decrease in Impurity) a través de todos los árboles.

Las características más importantes para diferenciar tumores malignos de benignos son:
- **worst concave points** — descripción de la forma del núcleo
- **worst perimeter** — tamaño del núcleo en su peor medición
- **mean concave points** — concavidades del núcleo
- **worst radius** — radio en su peor medición

---

### 7. Efecto del Número de Árboles

El accuracy se estabiliza aproximadamente a partir de 100–150 árboles. Agregar más árboles aumenta el costo computacional sin mejora significativa en rendimiento.

---

## Visualizaciones Generadas

| Archivo | Descripción |
|---------|-------------|
| `01_eda_overview.png` | Distribución de clases + boxplots clave |
| `02_feature_distributions.png` | Histogramas por diagnóstico (6 features) |
| `03_correlation_matrix.png` | Correlación de las primeras 10 características |
| `04_confusion_matrices.png` | Matrices de confusión (Árbol vs RF) |
| `05_roc_curves.png` | Curvas ROC con AUC para ambos modelos |
| `06_feature_importance_rf.png` | Top 15 características más importantes |
| `07_model_comparison.png` | Comparación de métricas Árbol vs RF |
| `08_n_estimators_effect.png` | Accuracy en función de n_estimators |

---

## Conceptos Clave

### Curva ROC y AUC

```
AUC = 1.0  → clasificador perfecto
AUC = 0.5  → clasificador aleatorio (diagonal)
AUC > 0.9  → excelente discriminación
```

La curva ROC muestra el trade-off entre Tasa de Verdaderos Positivos (Sensibilidad) y Tasa de Falsos Positivos.

### Sesgo-Varianza en Ensambles

| Propiedad | Árbol | RF |
|-----------|-------|-----|
| Sesgo | Bajo | Bajo (heredado) |
| Varianza | Alta | Baja (reducida por promedio) |
| Overfitting | Frecuente | Raro |

---

## Herramientas y Librerías

| Librería | Uso |
|----------|-----|
| **scikit-learn** | `RandomForestClassifier`, `DecisionTreeClassifier`, `GridSearchCV` |
| **sklearn.metrics** | `roc_auc_score`, `roc_curve`, `confusion_matrix` |
| **pandas** | Manejo del DataFrame |
| **numpy** | Operaciones numéricas |
| **matplotlib** | Visualización |
| **seaborn** | Heatmaps y estilo |

---

## Checklist de Requisitos

- División train/test con estratificación
- Árbol de decisión individual (baseline)
- Random Forest con GridSearchCV + StratifiedKFold
- OOB Score calculado
- Reporte de clasificación completo
- Matrices de confusión (Árbol vs RF)
- Curvas ROC con AUC para ambos modelos
- Importancia de características (Top 15)
- Comparación de métricas (gráfico)
- Efecto del número de árboles
- Cross-validation final

---

**FIN DE DOCUMENTACIÓN**
