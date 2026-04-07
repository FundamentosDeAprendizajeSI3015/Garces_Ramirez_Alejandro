# Ejercicio Práctico - Semana 9: Agrupamiento K-Means y DBSCAN (FIRE-UdeA)
## SI3015 - Fundamentos de Aprendizaje Automático

**Autor:** Alejandro Garcés Ramírez
**Fecha:** Marzo 2026
**Dataset:** FIRE-UdeA Sintético y Realista

---

## Descripción General

Este notebook implementa técnicas de **aprendizaje no supervisado** (clustering) sobre dos versiones del dataset FIRE-UdeA. Se comparan los algoritmos **K-Means** y **DBSCAN**, analizando sus fortalezas, limitaciones y la estructura inherente de los datos financieros universitarios.

---

## Ejecución del Código

### Requisitos
```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

### Abrir el notebook
```bash
jupyter notebook Agrupamiento_kmeans_dbscan_FIRE.ipynb
```

---

## Datasets Utilizados

### Dataset 1: FIRE-UdeA Sintético (500 registros)

| Característica | Valor |
|----------------|-------|
| **Registros** | 500 |
| **Características** | 7 variables financieras |
| **Valores faltantes** | Ninguno |
| **Generación** | Sintético con estructura controlada |

**Variables:** liquidez, días de efectivo, CFO, participación Ley 30, HHI de fuentes, gastos de personal, tendencia de ingresos.

### Dataset 2: FIRE-UdeA Realista (80 registros)

| Característica | Valor |
|----------------|-------|
| **Registros** | 80 (8 unidades × 10 años) |
| **Características** | 13 variables financieras |
| **Valores faltantes** | Sí (imputación por mediana) |
| **Nota** | Etiquetas excluidas del clustering |

---

## Análisis Realizado

### 1. K-Means

#### Proceso
1. **K=2 inicial:** Primera iteración con K=2 para exploración
2. **Método del Codo (Elbow Method):** Gráfico de inercia vs K para encontrar el K óptimo

```python
# Inercia = suma de distancias al cuadrado de cada punto a su centroide
inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
```

3. **K-Means óptimo:** Con el K identificado por el codo
4. **Análisis de Silhouette:** Validación de la cohesión interna de los clusters

```
Silhouette Score ∈ [-1, 1]
  → 1: clusters perfectamente separados
  → 0: puntos en el borde entre clusters
  → -1: puntos mal asignados
```

#### Supuestos de K-Means
- Clusters globulares (esféricos)
- Tamaños similares
- Varianzas similares
- Requiere especificar K a priori

### 2. DBSCAN

```python
dbscan = DBSCAN(eps=2.0, min_samples=5)  # Dataset 1
dbscan = DBSCAN(eps=4.0, min_samples=4)  # Dataset 2
```

#### Conceptos DBSCAN
- **ε (eps):** Radio de vecindad de un punto
- **min_samples:** Mínimo de puntos para considerar un punto como "core point"
- **Core points:** Puntos con al menos `min_samples` vecinos dentro de radio ε
- **Border points:** Alcanzables desde un core point, pero sin suficientes vecinos propios
- **Outliers (ruido):** Puntos no alcanzables → label = -1

#### Ventajas sobre K-Means
- No requiere especificar K
- Detecta clusters de forma arbitraria (no solo globular)
- Identifica outliers automáticamente

### 3. Visualización con PCA

**Importante:** PCA se usa **solo para visualización** — el clustering se realiza en el espacio de características completo.

```python
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
# Graficar puntos en 2D coloreados por cluster
```

---

## Comparación K-Means vs DBSCAN

| Característica | K-Means | DBSCAN |
|----------------|---------|--------|
| K requerido | Sí | No |
| Forma de clusters | Globular | Arbitraria |
| Detección outliers | No | Sí |
| Sensibilidad a escala | Alta | Alta |
| Complejidad | O(n·k·i) | O(n²) o O(n log n) |
| Resultado en FIRE | Clusters definidos | Muchos outliers (datos financieros uniformes) |

### Hallazgo Clave

Los datos financieros del FIRE-UdeA **no presentan clusters de densidad clara** en el espacio multidimensional. K-Means fuerza una partición pero DBSCAN identifica la mayoría como outliers cuando se aplica al Dataset 2 (80 registros, alta dimensionalidad). Esto sugiere que los datos se distribuyen de manera relativamente uniforme sin agrupaciones naturales densas.

---

## Métricas de Evaluación de Clustering

```
Silhouette Score:  Cohesión interna vs separación entre clusters
Inercia (K-Means): Suma de distancias al cuadrado al centroide
Davies-Bouldin:    Promedio de similitudes entre clusters (menor = mejor)
```

---

## Herramientas y Librerías

| Librería | Uso |
|----------|-----|
| **scikit-learn** | `KMeans`, `DBSCAN`, `PCA`, `silhouette_score` |
| **pandas** | Carga y manipulación |
| **numpy** | Operaciones numéricas |
| **matplotlib** | Visualización |
| **seaborn** | Estilo de gráficos |
| **jupyter** | Entorno de notebook interactivo |

---

## Checklist de Requisitos

- Carga y exploración de dos datasets FIRE-UdeA
- Imputación de valores faltantes (Dataset 2)
- Estandarización de características
- K-Means con K=2 inicial
- Método del Codo para K óptimo
- K-Means con K óptimo
- DBSCAN con parámetros ajustados
- PCA para visualización 2D
- Comparación K-Means vs DBSCAN
- Análisis de Silhouette Score

---

**FIN DE DOCUMENTACIÓN**
