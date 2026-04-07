# Ejercicio Práctico - Semana 3: Preprocesamiento de Datos (Fintech)
## SI3015 - Fundamentos de Aprendizaje Automático

**Autor:** Alejandro Garcés Ramírez
**Fecha:** Febrero 2026
**Dataset:** Dataset Fintech

---

## Descripción General

Este código implementa un pipeline completo de exploración y preprocesamiento de datos sobre un dataset del sector fintech. El enfoque central es que la **calidad del preprocesamiento determina la calidad del modelo**: sin datos bien preparados, cualquier algoritmo produce resultados deficientes.

---

## Ejecución del Código

### Requisitos
```bash
pip install numpy pandas scikit-learn matplotlib seaborn scipy
```

### Ejecutar el análisis
```bash
python lecture_3_Garces_Ramirez_Alejandro_fintech.py
```

---

## Análisis Realizado

### 1. Carga de Datos

- Lectura robusta de CSV con manejo de codificaciones (`utf-8`, `latin-1`, `cp1252`)
- Inspección inicial: forma del dataset, tipos de datos, primeras filas
- Reporte automático de valores faltantes y duplicados

### 2. Exploración de Datos

- **Análisis estructural:** dtypes, categorías vs numéricas
- **Valores faltantes:** porcentaje por columna, mapa de calor de NaN
- **Estadísticas descriptivas:** media, mediana, desviación estándar, cuartiles
- **Duplicados:** detección y manejo

### 3. Limpieza de Datos

#### Tratamiento de Outliers
- **Detección:** Método IQR (Rango Intercuartílico)
  ```
  Q1 = percentil 25
  Q3 = percentil 75
  IQR = Q3 - Q1
  Límite inferior = Q1 - 1.5 × IQR
  Límite superior = Q3 + 1.5 × IQR
  ```
- **Tratamiento:** Winsorización — limita valores extremos a los umbrales sin eliminarlos
- **Filosofía:** exploración antes de remoción automática

#### Valores Faltantes
- Impuación por mediana para variables numéricas (robusta a outliers)
- Variables categóricas tratadas con moda o categoría especial

### 4. Ingeniería de Características

- Creación de variables derivadas con significado financiero
- Transformaciones logarítmicas para variables sesgadas
- Encoding de variables categóricas

### 5. División y Escalado

```python
# División estratificada antes de escalar (PREVENCIÓN DE DATA LEAKAGE)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scaler se ajusta SOLO en entrenamiento
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # ← transform, no fit_transform
```

### 6. Salida

- Datasets preprocesados listos para modelado (`X_train_scaled`, `X_test_scaled`)
- Informe de transformaciones aplicadas

---

## Conceptos Clave

### ¿Por qué Winsorización en lugar de eliminación?

La eliminación de outliers puede introducir sesgo si los valores extremos son legítimos (p.e., transacciones de alto valor en fintech). La Winsorización preserva el registro pero limita su influencia:

```
Valor original: 50,000 → límite superior: 10,000 → valor winsorizado: 10,000
```

### Prevención de Data Leakage

El error más común en preprocesamiento es calcular estadísticas de normalización sobre todo el dataset (incluyendo test):

```python
# MAL: el scaler "ve" los datos de test antes de predecir
scaler.fit_transform(X)  # ← data leakage

# BIEN: el test nunca informa la transformación
scaler.fit(X_train)
scaler.transform(X_test)
```

---

## Herramientas y Librerías

| Librería | Uso |
|----------|-----|
| **pandas** | Carga, limpieza y manipulación |
| **numpy** | Operaciones numéricas |
| **scikit-learn** | Preprocesamiento, escalado, splits |
| **matplotlib** | Visualización |
| **seaborn** | Visualización estadística |
| **scipy** | Estadísticos para detección de outliers |

---

## Filosofía del Módulo

> "El 80% del trabajo en ML es preprocesamiento. Un modelo mediocre con datos bien preparados supera a un modelo sofisticado con datos sucios."

- **Separar** exploración de transformación
- **Documentar** cada decisión de limpieza
- **Preservar** los datos originales (trabajar sobre copias)
- **Verificar** antes de eliminar (outliers pueden ser señal, no ruido)

---

**FIN DE DOCUMENTACIÓN**
