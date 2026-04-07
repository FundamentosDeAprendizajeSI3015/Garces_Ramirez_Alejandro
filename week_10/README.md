# Ejercicio Práctico - Semana 10: Validación de Etiquetas y Clustering Avanzado
## SI3015 - Fundamentos de Aprendizaje Automático

**Autor:** Alejandro Garcés Ramírez
**Fecha:** Abril 2026
**Dataset:** FIRE-UdeA Realista (validación de etiquetas)

---

## Descripción General

Este módulo implementa un pipeline avanzado de **validación de etiquetas mediante clustering no supervisado**. El objetivo es evaluar si las etiquetas originales del dataset son consistentes con la estructura natural de los datos, usando un ensamble de algoritmos de clustering como validación externa.

---

## Ejecución del Código

### Requisitos
```bash
pip install numpy pandas scikit-learn matplotlib seaborn scipy
```

### Ejecutar el análisis
```bash
python label_validation_analysis.py
```

Las visualizaciones se generan en `graficas_clustering/`.

---

## Descripción del Problema

### ¿Por qué validar etiquetas?

Las etiquetas de un dataset pueden ser:
- **Incorrectas:** errores de anotación humana
- **Inconsistentes:** criterios de etiquetado no uniformes
- **Ruidosas:** condiciones ambiguas en el borde de decisión

El clustering permite verificar si la estructura natural de los datos corresponde con las etiquetas asignadas. Si los clusters no coinciden con las clases, puede indicar problemas en el proceso de etiquetado.

---

## Pipeline de Análisis

```
Datos originales
      │
      ▼
Separar etiquetas (y) ──► guardar para validación final
      │
      ▼
Preprocesamiento de X
  • Imputar NaN (mediana)
  • StandardScaler
      │
      ▼
Características polinomiales (grado 2)
  • Captura interacciones entre variables financieras
      │
      ▼
Ensamble de clustering (sobre X aumentado)
  • K-Means
  • DBSCAN
  • Clustering jerárquico (Ward)
  • [Otros algoritmos]
      │
      ▼
Validación interna        Validación externa
  • Silhouette Score         • ARI vs etiquetas originales
  • Davies-Bouldin           • NMI vs etiquetas originales
      │
      ▼
Visualizaciones 2D (PCA)
```

---

## Componentes Clave

### 1. Separación de Etiquetas (Anti-Filtración)

```python
# Las etiquetas se extraen ANTES del clustering
y = df['label'].copy()
X = df.drop('label', axis=1)

# Solo se reintroducen al final para validación externa
# El clustering trabaja completamente sin supervisión
```

**Principio fundamental:** El clustering no tiene acceso a las etiquetas durante el proceso. Si los clusters coinciden con las clases, es evidencia de que las etiquetas reflejan la estructura real de los datos.

### 2. Características Polinomiales

```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)
# p=7 características → 7 + 21 interacciones = 28 features
```

Las interacciones entre variables financieras pueden revelar patrones más complejos:
- `liquidez × CFO`: combinación de solvencia y generación de caja
- `HHI × Ley30`: concentración de ingresos en fuente dependiente del Estado

### 3. Ensamble de Clustering

Múltiples algoritmos con diferentes supuestos proporcionan un consenso más robusto:

| Algoritmo | Supuesto de clusters |
|-----------|----------------------|
| K-Means | Globular, tamaño similar |
| DBSCAN | Densidad arbitraria |
| Jerárquico (Ward) | Minimiza varianza intra-cluster |

### 4. Métricas de Validación

#### Internas (no necesitan etiquetas)
```
Silhouette Score ∈ [-1, 1]:
  → Alta: buena cohesión y separación
  → Baja: clusters solapados o mal definidos
```

#### Externas (comparan con etiquetas originales)
```
ARI (Adjusted Rand Index) ∈ [-1, 1]:
  → 1: coincidencia perfecta con etiquetas
  → 0: coincidencia aleatoria
  → Negativo: peor que aleatorio

NMI (Normalized Mutual Information) ∈ [0, 1]:
  → 1: información mutua perfecta
  → 0: independientes
```

---

## Visualizaciones Generadas

| Archivo | Descripción |
|---------|-------------|
| `graficas_clustering/kmeans_pca.png` | Clusters K-Means proyectados en 2D |
| `graficas_clustering/dbscan_pca.png` | Clusters DBSCAN (con outliers) |
| `graficas_clustering/jerarquico_pca.png` | Clusters jerárquicos |
| `graficas_clustering/silhouette_plot.png` | Análisis de Silhouette por muestra |
| `graficas_clustering/label_comparison.png` | Clusters vs etiquetas originales |
| `graficas_clustering/metrics_summary.png` | Resumen de métricas de validación |

---

## Interpretación de Resultados

### ARI alto (> 0.7)
Los clusters no supervisados coinciden en gran medida con las etiquetas originales. Las etiquetas son **consistentes** con la estructura de los datos.

### ARI bajo (< 0.3)
Los clusters difieren significativamente de las etiquetas. Posibles causas:
1. Las etiquetas no reflejan la estructura natural de los datos
2. Los criterios de etiquetado son subjetivos o mixtos
3. Los datos en los bordes de clase son ambiguos

---

## Conceptos Clave

### Diferencia vs Semana 9

| Semana 9 | Semana 10 |
|----------|-----------|
| Clustering exploratorio | Clustering como validación |
| K-Means y DBSCAN básicos | Ensamble de clustering |
| Sin etiquetas disponibles | Etiquetas usadas solo al final |
| PCA para visualización | Características polinomiales + PCA |

---

## Herramientas y Librerías

| Librería | Uso |
|----------|-----|
| **scikit-learn** | `KMeans`, `DBSCAN`, `AgglomerativeClustering`, `PolynomialFeatures` |
| **sklearn.metrics** | `silhouette_score`, `adjusted_rand_score`, `normalized_mutual_info_score` |
| **pandas** | Carga y manipulación |
| **numpy** | Operaciones numéricas |
| **matplotlib** | Visualización |
| **seaborn** | Estilo de gráficos |

---

## Checklist de Requisitos

- Separación previa de etiquetas (sin data leakage al clustering)
- Preprocesamiento: imputación + estandarización
- Características polinomiales grado 2
- Ensamble de clustering (K-Means + DBSCAN + Jerárquico)
- Métricas internas: Silhouette Score
- Métricas externas: ARI y NMI vs etiquetas originales
- Visualizaciones 2D con PCA por algoritmo
- Interpretación de consistencia de etiquetas

---

**FIN DE DOCUMENTACIÓN**
