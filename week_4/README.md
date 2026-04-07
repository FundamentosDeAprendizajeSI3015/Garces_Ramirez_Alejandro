# Ejercicio Práctico - Semana 4: Limpieza y Exploración de Datos (Movies)
## SI3015 - Fundamentos de Aprendizaje Automático

**Autor:** Alejandro Garcés Ramírez
**Fecha:** Febrero 2026
**Dataset:** IMDB Movies (movies.csv — 10,000 películas)

---

## Descripción General

Este código implementa un pipeline completo de limpieza y transformación de datos sobre un dataset real de películas IMDB con múltiples problemas de calidad. El objetivo es preparar el dataset para su uso posterior en modelos de aprendizaje automático.

---

## Ejecución del Código

### Requisitos
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### Ejecutar el análisis
```bash
python solucion.py
```

---

## Dataset

| Característica | Valor |
|----------------|-------|
| **Archivo** | `movies.csv` |
| **Filas originales** | ~10,000 películas |
| **Columnas** | MOVIES, YEAR, GENRE, RATING, VOTES, RunTime, Gross |
| **Problemas** | Tipos mixtos, símbolos de moneda, fechas en rangos, abreviaciones |

---

## Análisis Realizado

### 1. Problemas de Calidad Encontrados

| Columna | Problema | Ejemplo |
|---------|----------|---------|
| `VOTES` | Separador de miles por coma | `"1,234,567"` |
| `Gross` | Símbolo de moneda + abreviaciones | `"$45.3M"`, `"$2.1B"` |
| `YEAR` | Rangos de años | `"2010–2022"` |
| `RunTime` | Texto mezclado | `"142 min"` |
| Múltiples | Valores faltantes (NaN) | — |

### 2. Conversión de Tipos

```python
# Eliminar comas en miles
df['VOTES'] = df['VOTES'].str.replace(',', '').astype(float)

# Parsear monedas: $45.3M → 45,300,000
def parse_currency(val):
    val = str(val).replace('$', '').strip()
    if val.endswith('B'): return float(val[:-1]) * 1e9
    if val.endswith('M'): return float(val[:-1]) * 1e6
    if val.endswith('K'): return float(val[:-1]) * 1e3
    return float(val)

# Extraer año de rangos: "2010–2022" → 2010
df['YEAR'] = df['YEAR'].str.split('–').str[0].str.extract(r'(\d{4})')
```

### 3. Estadísticas Descriptivas

Para cada variable numérica:
- Media, Mediana, Moda
- Desviación estándar y varianza
- Rango intercuartílico (IQR)
- Mínimo y máximo

### 4. Detección de Outliers (Método IQR)

```
Límite inferior = Q1 - 1.5 × IQR
Límite superior = Q3 + 1.5 × IQR
```

Los outliers son **identificados y reportados** pero no eliminados automáticamente — la decisión de tratamiento depende del contexto.

### 5. Codificación de Variables

#### Label Encoding (variables ordinales)
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['GENRE_encoded'] = le.fit_transform(df['GENRE'])
```

#### One-Hot Encoding (variables nominales)
```python
genre_dummies = pd.get_dummies(df['GENRE'], prefix='genre')
df = pd.concat([df, genre_dummies], axis=1)
```

### 6. Escalado de Características

| Método | Fórmula | Rango resultante | Cuándo usar |
|--------|---------|-----------------|-------------|
| **MinMaxScaler** | `(x - min) / (max - min)` | [0, 1] | Sin outliers extremos |
| **StandardScaler** | `(x - μ) / σ` | Media=0, Std=1 | Con distribución aproximadamente normal |

### 7. Análisis de Correlaciones

- Matriz de correlación de Pearson entre variables numéricas
- Identificación de relaciones entre RATING, VOTES y Gross
- Heatmap para visualización

### 8. Dataset de Salida

El script genera un CSV limpio con:
- Tipos de datos corregidos
- Valores faltantes imputados o eliminados según criterio
- Variables codificadas y escaladas

---

## Decisiones de Diseño

### ¿Por qué no eliminar automáticamente los outliers?

En películas, un ingreso de $2.5B (Avatar) es legítimo aunque sea un outlier estadístico. La eliminación automática borraría información valiosa. La estrategia correcta es:

1. **Identificar** con IQR
2. **Investigar** el contexto de cada outlier
3. **Decidir** basado en el dominio: ¿error de datos o valor real?

---

## Herramientas y Librerías

| Librería | Uso |
|----------|-----|
| **pandas** | Carga, limpieza y manipulación |
| **numpy** | Operaciones numéricas |
| **scikit-learn** | LabelEncoder, MinMaxScaler, StandardScaler |
| **matplotlib** | Visualización |
| **seaborn** | Heatmaps y distribuciones |

---

**FIN DE DOCUMENTACIÓN**
