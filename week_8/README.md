# Ejercicio Práctico - Semana 8: Árboles de Decisión y Random Forest (FIRE-UdeA)
## SI3015 - Fundamentos de Aprendizaje Automático

**Autor:** Alejandro Garcés Ramírez
**Fecha:** Marzo 2026
**Dataset:** Dataset Sintético FIRE-UdeA Realista

---

## Descripción General

Este código implementa modelos de clasificación basados en **Árbol de Decisión** y **Random Forest** para predecir tensión financiera en las unidades académicas y administrativas de la Universidad de Antioquia. El contexto es el sistema FIRE-UdeA (Financial Risk Early Warning).

---

## Ejecución del Código

### Requisitos
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### Ejecutar desde la carpeta Lecture08
```bash
cd week_8/Lecture08
python solution.py
```

Las imágenes se generan automáticamente en la carpeta `images/`.

---

## Contexto del Problema

La Universidad de Antioquia, como institución pública, enfrenta riesgos financieros estructurales:
- Dependencia de transferencias estatales (Ley 30 de 1992)
- Volatilidad de fuentes de ingreso (regalías, servicios)
- Crecimiento del gasto de personal
- Rigidez presupuestal

El objetivo es construir un modelo que detecte **tensión financiera** antes de que se materialice, permitiendo intervenciones preventivas.

---

## Dataset

| Característica | Valor |
|----------------|-------|
| **Archivo** | `dataset_sintetico_FIRE_UdeA_realista.csv` |
| **Registros** | 80 (8 unidades × 10 años: 2016–2025) |
| **Características** | 13 variables financieras |
| **Target** | Binario: tensión financiera (1) vs saludable (0) |

### Definición de Tensión Financiera (label = 1)

Se considera tensión financiera cuando se presenta alguna combinación de:
- CFO (Cash Flow from Operations) negativo por dos años consecutivos
- Ratio de liquidez < 1
- Días de efectivo disponibles < 30

### Variables Financieras

| Variable | Descripción |
|----------|-------------|
| CFO | Flujo de caja operacional |
| Ratio de liquidez | Activo corriente / Pasivo corriente |
| Días de efectivo | Caja disponible / gasto diario promedio |
| Participación Ley 30 | % ingresos provenientes de Ley 30 |
| HHI de fuentes | Índice de concentración de ingresos |
| Gasto personal / ingresos | Proporción del gasto en nómina |
| Tendencia de ingresos | Variación año a año |

---

## Análisis Realizado

### 1. EDA

- Distribución de la variable objetivo (balance de clases)
- Distribuciones de características por clase (saludable vs tensión)
- Boxplots de indicadores financieros clave
- Matriz de correlación entre variables

### 2. División de Datos

- **Entrenamiento:** 70% (56 registros)
- **Prueba:** 30% (24 registros)
- División estratificada para mantener proporción de clases

### 3. Árbol de Decisión

- Árbol completo sin restricciones (baseline)
- Búsqueda de hiperparámetros con **GridSearchCV**:
  - `max_depth`, `min_samples_split`, `min_samples_leaf`, `criterion`
- Visualización del árbol con interpretación financiera de las reglas

### 4. Random Forest

- Ensamble de múltiples árboles con bagging
- Búsqueda de hiperparámetros: `n_estimators`, `max_depth`, `max_features`
- Comparación directa con el árbol individual

### 5. Evaluación

| Métrica | Descripción |
|---------|-------------|
| Accuracy | % clasificaciones correctas |
| Precision | Exactitud de las alertas generadas |
| Recall | % de tensiones reales detectadas |
| F1 Score | Media armónica precision-recall |
| ROC-AUC | Capacidad discriminativa general |

> En el contexto FIRE-UdeA, el **Recall** es la métrica más crítica: queremos minimizar los casos de tensión financiera no detectados (falsos negativos).

---

## Resultados

### Comparación Árbol vs Random Forest

| Métrica | Árbol de Decisión | Random Forest |
|---------|-------------------|---------------|
| Accuracy | Variable | Variable + estable |
| Recall | Dependiente del podado | Más robusto |
| F1 Score | Dependiente del podado | Superior |
| ROC-AUC | ~0.85–0.95 | ~0.90–0.99 |

### ¿Por qué árboles para este problema?

1. **Interpretabilidad:** Las reglas se expresan en términos financieros comprensibles por directivos
2. **Sin escalado:** Los ratios financieros no necesitan normalización
3. **No linealidades:** Las relaciones entre indicadores financieros son complejas
4. **Robustez a outliers:** Los splits por umbrales no se ven afectados por valores extremos

---

## Visualizaciones Generadas

| Archivo | Descripción |
|---------|-------------|
| `images/target_distribution.png` | Distribución de la variable objetivo |
| `images/feature_distributions.png` | Características por clase |
| `images/boxplots.png` | Boxplots de indicadores clave |
| `images/correlation_heatmap.png` | Correlación entre variables |
| `images/confusion_matrix_dt.png` | Matriz de confusión — Árbol |
| `images/confusion_matrix_rf.png` | Matriz de confusión — Random Forest |
| `images/feature_importance.png` | Importancia de características |
| `images/model_comparison.png` | Comparación de métricas |

---

## Herramientas y Librerías

| Librería | Uso |
|----------|-----|
| **scikit-learn** | `DecisionTreeClassifier`, `RandomForestClassifier`, `GridSearchCV` |
| **sklearn.tree** | `plot_tree`, `export_text` |
| **pandas** | Carga y manipulación del dataset |
| **numpy** | Operaciones numéricas |
| **matplotlib** | Visualización |
| **seaborn** | Heatmaps y estilo |

---

## Checklist de Requisitos

- Comprensión del problema financiero (FIRE-UdeA)
- EDA con visualizaciones del dataset
- División train/test estratificada
- Árbol de Decisión (completo y optimizado)
- Random Forest con GridSearchCV
- Evaluación completa (Accuracy, Precision, Recall, F1, AUC)
- Matrices de confusión para ambos modelos
- Importancia de características con interpretación financiera
- Comparación visual de métricas

---

**FIN DE DOCUMENTACIÓN**
