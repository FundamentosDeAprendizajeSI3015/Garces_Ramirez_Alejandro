# %% [markdown]
# # FIRE-UdeA: Modelo Predictivo de Riesgo Financiero
# ## Clasificación con Árboles de Decisión y Random Forest
# 
# ---
# 
# **Objetivo:** Construir un modelo de clasificación basado en Árbol de Decisión y Random Forest (a ver cual sale mejor) para estimar la probabilidad de tensión financiera (tensión de caja en t+1) en las unidades académicas y administrativas de la Universidad de Antioquia, comparando los resultados con un modelo previo de Gradient Boosting.

# %% [markdown]
# ## 1. Comprensión del Problema
# 
# ### 1.1 Contexto institucional
# 
# La Universidad de Antioquia, como institución pública de educación superior, enfrenta desafíos estructurales de financiamiento que afectan su sostenibilidad operativa. La dependencia de transferencias del Estado (Ley 30 de 1992), la volatilidad de fuentes como regalías y venta de servicios, el crecimiento sostenido de los gastos de personal y la rigidez presupuestal configuran un escenario de riesgo financiero latente.
# 
# ### 1.2 ¿Por qué un modelo predictivo?
# 
# La detección temprana de tensión de caja permite a la administración universitaria:
# - Anticipar periodos de iliquidez antes de que se materialicen.
# - Priorizar fuentes de ingreso estratégicas.
# - Dimensionar el impacto del gasto de personal sobre la sostenibilidad.
# - Facilitar la toma de decisiones basada en evidencia cuantitativa.
# 
# ### 1.3 Definición de la variable objetivo
# 
# Se define **tensión financiera** (label = 1) cuando se presenta alguna combinación de las siguientes condiciones:
# - CFO (Cash Flow from Operations) negativo por dos años consecutivos.
# - Ratio de liquidez inferior a 1.
# - Días de efectivo disponibles menores a 30.
# 
# Estas condiciones capturan deterioros simultáneos en liquidez, generación de caja y capacidad operativa.
# 
# ### 1.4 ¿Por qué árboles de decisión y Random Forest?
# 
# Los modelos basados en árboles son especialmente adecuados para este caso porque:
# - **Interpretabilidad:** Permiten visualizar las reglas de decisión en lenguaje financiero.
# - **No requieren escalado:** Son invariantes a transformaciones monótonas de las variables.
# - **Capturan no linealidades:** Las relaciones entre ratios financieros no siempre son lineales.
# - **Robustez a outliers:** Los splits se basan en umbrales, no en distancias.
# - **Random Forest** agrega estabilidad mediante el ensamble de múltiples árboles, reduciendo varianza y overfitting.

# %% [markdown]
# ## 2. Carga de Datos (LOAD DATA)

# %%
# ==============================
# 1. LOAD DATA
# ==============================
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo para guardar figuras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     GridSearchCV, StratifiedKFold)
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve,
                             f1_score, precision_score, recall_score)
import warnings
warnings.filterwarnings('ignore')

# Crear carpeta de imágenes si no existe
os.makedirs('images', exist_ok=True)

# Configuración visual
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
sns.set_style('whitegrid')

print("Librerías cargadas correctamente.")
print(f"Pandas: {pd.__version__}")
print(f"NumPy: {np.__version__}")

# %%
# Cargar dataset
df = pd.read_csv('dataset_sintetico_FIRE_UdeA_realista.csv')
print(f"Dataset cargado exitosamente: {df.shape[0]} filas x {df.shape[1]} columnas")
print(f"\nColumnas disponibles:")
for i, col in enumerate(df.columns):
    print(f"  {i+1:2d}. {col}")

# %% [markdown]
# ## 3. Análisis Exploratorio de Datos (BASIC EDA)
# 
# ### 3.1 Inspección inicial

# %%
# ==============================
# 2. BASIC EDA
# ==============================

# Vista previa
print("=" * 60)
print("PRIMERAS FILAS DEL DATASET")
print("=" * 60)
df.head()

# %%
# Información del dataset
print("=" * 60)
print("INFORMACIÓN DEL DATASET")
print("=" * 60)
df.info()

# %%
# Estadísticas descriptivas
print("=" * 60)
print("ESTADÍSTICAS DESCRIPTIVAS")
print("=" * 60)
df.describe().round(4)

# %% [markdown]
# ### 3.2 Comprensión del dataset
# 
# | Aspecto | Descripción |
# |---------|-------------|
# | **Tamaño** | 80 filas × 16 columnas |
# | **Unidades** | 8 unidades académicas/administrativas de la UdeA |
# | **Periodo** | 2016–2025 (10 años por unidad) |
# | **Variable objetivo** | `label` (0 = sin tensión, 1 = tensión financiera) |
# | **Variables predictoras** | 13 variables financieras continuas |
# | **Identificadores** | `anio` (año), `unidad` (nombre de la dependencia) |
# 
# **Variables financieras del dataset:**
# - `ingresos_totales`: Ingresos totales de la unidad.
# - `gastos_personal`: Gasto en nómina y personal.
# - `liquidez`: Ratio corriente (activos corrientes / pasivos corrientes).
# - `dias_efectivo`: Días de efectivo disponible.
# - `cfo`: Cash Flow from Operations (flujo de caja operativo).
# - `participacion_ley30`: Proporción de ingresos provenientes de la Ley 30.
# - `participacion_regalias`: Proporción de ingresos por regalías.
# - `participacion_servicios`: Proporción de ingresos por venta de servicios.
# - `participacion_matriculas`: Proporción de ingresos por matrículas.
# - `hhi_fuentes`: Índice Herfindahl-Hirschman de concentración de fuentes de ingreso.
# - `endeudamiento`: Ratio de endeudamiento.
# - `tendencia_ingresos`: Tasa de cambio porcentual de ingresos año a año.
# - `gp_ratio`: Ratio gastos de personal / ingresos totales.

# %% [markdown]
# ### 3.3 Valores nulos

# %%
# Valores nulos
print("=" * 60)
print("VALORES NULOS POR VARIABLE")
print("=" * 60)
nulos = df.isnull().sum()
nulos_pct = (df.isnull().sum() / len(df) * 100).round(2)
nulos_df = pd.DataFrame({'Nulos': nulos, 'Porcentaje (%)': nulos_pct})
nulos_df = nulos_df[nulos_df['Nulos'] > 0]
print(nulos_df)
print(f"\nTotal de valores nulos en el dataset: {df.isnull().sum().sum()}")
print(f"Porcentaje total: {(df.isnull().sum().sum() / df.size * 100):.2f}%")

# %% [markdown]
# **Interpretación:** Se observan valores nulos en 7 variables, siendo `endeudamiento` la más afectada (10%). Estos nulos son plausibles en datos financieros institucionales donde algunas unidades no reportan ciertas métricas en ciertos periodos. Se imputarán con la mediana, que es robusta frente a outliers.

# %% [markdown]
# ### 3.4 Duplicados

# %%
# Duplicados
print("=" * 60)
print("ANÁLISIS DE DUPLICADOS")
print("=" * 60)
duplicados = df.duplicated().sum()
print(f"Filas duplicadas: {duplicados}")
if duplicados == 0:
    print("No se encontraron filas duplicadas. El dataset es limpio en este aspecto.")

# %% [markdown]
# ### 3.5 Distribución de la variable objetivo

# %%
# Distribución de la variable objetivo
print("=" * 60)
print("DISTRIBUCIÓN DE LA VARIABLE OBJETIVO")
print("=" * 60)
print(df['label'].value_counts())
print(f"\nProporción clase 0 (sin tensión): {(df['label']==0).mean():.2%}")
print(f"Proporción clase 1 (tensión):      {(df['label']==1).mean():.2%}")

fig, ax = plt.subplots(figsize=(6, 4))
colors = ['#4C72B0', '#DD8452']
df['label'].value_counts().plot(kind='bar', color=colors, ax=ax, edgecolor='black')

ax.set_title('Distribución de la Variable Objetivo', fontsize=14, fontweight='bold')
ax.set_xlabel('Label (0 = Sin tensión, 1 = Tensión)')
ax.set_ylabel('Frecuencia')
ax.set_xticklabels(['0 - Sin tensión', '1 - Tensión'], rotation=0)

for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width()/2., p.get_height()),
                ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('images/01_distribucion_variable_objetivo.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figura guardada: images/01_distribucion_variable_objetivo.png")

# %% [markdown]
# **Interpretación:** Las clases están relativamente balanceadas (52.5% vs 47.5%), lo cual es favorable para el entrenamiento. No será necesario aplicar técnicas agresivas de rebalanceo como SMOTE, aunque se utilizará `stratify` en la partición para mantener las proporciones.

# %% [markdown]
# ### 3.6 Histogramas por variable y clase

# %%
# Histogramas por variable y clase
feature_cols = [c for c in df.columns if c not in ['anio', 'unidad', 'label']]

fig, axes = plt.subplots(4, 4, figsize=(16, 14))
axes = axes.flatten()

for i, col in enumerate(feature_cols):
    ax = axes[i]
    df[df['label']==0][col].hist(bins=15, alpha=0.6, color='#4C72B0', label='Sin tensión', ax=ax)
    df[df['label']==1][col].hist(bins=15, alpha=0.6, color='#DD8452', label='Tensión', ax=ax)
    ax.set_title(col, fontsize=9, fontweight='bold')
    ax.legend(fontsize=7)

for j in range(len(feature_cols), len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Distribución de Variables por Clase', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('images/02_histogramas_por_clase.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figura guardada: images/02_histogramas_por_clase.png")

# %% [markdown]
# **Interpretación de los histogramas:**
# - **`cfo`**: Los casos de tensión se concentran en valores bajos o negativos de flujo de caja operativo, confirmando que el CFO es una señal clave de deterioro.
# - **`dias_efectivo`**: Los periodos de tensión tienden a tener menos días de efectivo disponible.
# - **`gp_ratio`**: Un ratio alto de gastos de personal sobre ingresos se asocia con mayor tensión, reflejando la presión que ejerce la nómina sobre la sostenibilidad.
# - **`liquidez`**: Los valores más bajos de liquidez se asocian consistentemente con tensión financiera.
# - **`endeudamiento`**: Mayor endeudamiento aparece más frecuentemente en periodos de tensión.

# %% [markdown]
# ### 3.7 Boxplots por variable y clase

# %%
# Boxplots por variable y clase
fig, axes = plt.subplots(4, 4, figsize=(16, 14))
axes = axes.flatten()

for i, col in enumerate(feature_cols):
    ax = axes[i]
    data_bp = pd.DataFrame({'valor': df[col], 'label': df['label'].astype(str)})
    sns.boxplot(x='label', y='valor', data=data_bp, ax=ax, palette=['#4C72B0', '#DD8452'])
    ax.set_title(col, fontsize=9, fontweight='bold')
    ax.set_xlabel('')

for j in range(len(feature_cols), len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Boxplots de Variables por Clase', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('images/03_boxplots_por_clase.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figura guardada: images/03_boxplots_por_clase.png")

# %% [markdown]
# **Interpretación de los boxplots:**
# - Las variables con mayor separación entre clases (y por tanto mayor potencial predictivo) son: `cfo`, `gp_ratio`, `dias_efectivo`, `liquidez` y `endeudamiento`.
# - `ingresos_totales` y `gastos_personal` presentan gran dispersión pero menor separación entre clases, lo cual sugiere que los ratios derivados son más informativos que los montos absolutos.
# - `participacion_regalias` presenta poca variación y escasa diferenciación entre clases.

# %% [markdown]
# ### 3.8 Matriz de correlación

# %%
# Matriz de correlación
df_numeric = df[feature_cols + ['label']].copy()
for col in feature_cols:
    df_numeric[col] = df_numeric[col].fillna(df_numeric[col].median())

corr = df_numeric.corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax,
            square=True, linewidths=0.5, mask=mask)
ax.set_title('Matriz de Correlación', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('images/04_matriz_correlacion.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figura guardada: images/04_matriz_correlacion.png")

# Correlaciones con la variable objetivo
print("\n" + "=" * 60)
print("CORRELACIÓN CON LA VARIABLE OBJETIVO (label)")
print("=" * 60)
corr_label = corr['label'].drop('label').sort_values(ascending=False)
print(corr_label.round(4).to_string())

# %% [markdown]
# **Interpretación de la matriz de correlación:**
# 
# **Correlaciones más fuertes con tensión financiera (label):**
# - `gp_ratio` (positiva): A mayor proporción de gasto de personal, mayor riesgo de tensión. Esta es la señal más clara del dataset.
# - `endeudamiento` (positiva): Mayor endeudamiento se asocia con mayor tensión.
# - `cfo` (negativa): A menor flujo de caja operativo, mayor probabilidad de tensión.
# - `dias_efectivo` (negativa): Menos días de efectivo = mayor riesgo.
# - `liquidez` (negativa): Menor liquidez = mayor vulnerabilidad.
# 
# **Correlaciones entre predictores:**
# - `ingresos_totales` y `gastos_personal` están altamente correlacionados (esperado, pues el gasto crece con el tamaño de la unidad). Esto sugiere que ambas variables son parcialmente redundantes, pero `gp_ratio` captura mejor la relación relevante.
# - Las variables de participación por fuente tienen correlaciones moderadas entre sí, reflejando que son proporciones que suman aproximadamente 1.
# 
# **Conclusión del EDA:** Las variables con mayor poder discriminante para detectar tensión financiera son `cfo`, `gp_ratio`, `dias_efectivo`, `liquidez` y `endeudamiento`. Estos hallazgos son coherentes con la teoría financiera institucional.

# %% [markdown]
# ## 4. Limpieza y Preparación de Datos (CLEANING & TRANSFORMATION)

# %%
# ==============================
# 3. CLEANING & TRANSFORMATION
# ==============================

# Copia de trabajo
df_model = df.copy()

# Definir columnas de features (excluir identificadores y target)
feature_cols = [c for c in df.columns if c not in ['anio', 'unidad', 'label']]
print(f"Variables predictoras ({len(feature_cols)}):")
for col in feature_cols:
    print(f"  - {col}")

# Imputación de nulos con la mediana
# Justificación: la mediana es robusta a outliers y preserva la distribución central
print("\n--- Imputación de valores nulos con mediana ---")
for col in feature_cols:
    n_null = df_model[col].isnull().sum()
    if n_null > 0:
        median_val = df_model[col].median()
        df_model[col] = df_model[col].fillna(median_val)
        print(f"  {col}: {n_null} nulos → imputados con mediana = {median_val:.4f}")

# Verificar que no quedan nulos
assert df_model[feature_cols].isnull().sum().sum() == 0, "Aún hay nulos!"
print("\n✓ Todos los valores nulos han sido imputados correctamente.")

# %% [markdown]
# ### 4.1 Decisiones de preprocesamiento
# 
# | Decisión | Justificación |
# |----------|---------------|
# | **Imputación con mediana** | Robusta a outliers, preserva la distribución. Apropiada para datos financieros con posibles valores extremos. |
# | **No se aplica escalado** | Los árboles de decisión y Random Forest son invariantes a escalas. Los splits se determinan por umbrales, no por distancias. |
# | **No se codifican categóricas** | `anio` y `unidad` se excluyen como identificadores, no como predictores. Incluirlos podría causar data leakage temporal. |
# | **No se aplica SMOTE** | Las clases están suficientemente balanceadas (52.5% vs 47.5%). Se usa `stratify` en el split. |
# | **Se excluyen `anio` y `unidad`** | Son identificadores de contexto. Incluir `anio` como predictor introduciría una tendencia temporal artificial; `unidad` como categórica con 8 niveles en 80 filas causaría sobreajuste. |
# 
# ### 4.2 Prevención de data leakage
# 
# Se excluyen `anio` y `unidad` para evitar que el modelo memorice patrones específicos de tiempo o de unidad. La variable `label` se construyó a partir de condiciones sobre `cfo`, `liquidez` y `dias_efectivo`, por lo que estas variables están parcialmente correlacionadas con el target por diseño. Sin embargo, esto es intencional: el modelo debe aprender a detectar las combinaciones de indicadores que predicen tensión, no las condiciones individuales que la definen.

# %% [markdown]
# ## 5. Definición de Variables y Partición (FEATURES & TARGET / TRAIN-TEST SPLIT)

# %%
# ==============================
# 4. DEFINE FEATURES & TARGET
# ==============================
X = df_model[feature_cols].values
y = df_model['label'].values

print(f"Matriz de features X: {X.shape}")
print(f"Vector objetivo y:    {y.shape}")
print(f"\nDistribución de clases:")
print(f"  Clase 0 (sin tensión): {(y==0).sum()} ({(y==0).mean():.1%})")
print(f"  Clase 1 (tensión):     {(y==1).sum()} ({(y==1).mean():.1%})")

# %%
# ==============================
# 5. TRAIN / TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y  # Mantener proporciones de clase
)

print(f"Conjunto de entrenamiento: {X_train.shape[0]} muestras")
print(f"Conjunto de prueba:        {X_test.shape[0]} muestras")
print(f"\nDistribución en train: 0→{(y_train==0).sum()}, 1→{(y_train==1).sum()}")
print(f"Distribución en test:  0→{(y_test==0).sum()}, 1→{(y_test==1).sum()}")

# %% [markdown]
# **Estrategia de validación:** Se utiliza un split 75/25 con estratificación. Adicionalmente, se empleará validación cruzada estratificada con 5 folds para evaluar la estabilidad de los modelos. Se opta por StratifiedKFold sobre TimeSeriesSplit dado que el dataset combina múltiples unidades organizacionales y el objetivo es capturar patrones financieros generales, no secuencias temporales específicas.

# %% [markdown]
# ## 6. Modelo 1: Árbol de Decisión (Decision Tree)
# 
# ### 6.1 **Ventajas para FIRE-UdeA:**
# - Genera reglas interpretables: por ejemplo, "si CFO < X y liquidez < Y → tensión".
# - Permite identificar umbrales financieros críticos.
# - No requiere normalización de datos.
# 
# **Riesgos:**
# - Alta tendencia al **overfitting** si no se controla la profundidad.
# - Inestabilidad: pequeños cambios en los datos pueden generar árboles muy diferentes.
# - Puede no capturar bien interacciones complejas entre múltiples variables.

# %% [markdown]
# ### 6.2 Búsqueda de hiperparámetros (GridSearchCV)

# %%
# ==============================
# 6. DECISION TREE
# ==============================

# GridSearchCV para encontrar los mejores hiperparámetros
dt_params = {
    'max_depth': [2, 3, 4, 5, 6, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 3, 5],
    'criterion': ['gini', 'entropy']
}

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_params,
    cv=cv_strategy,
    scoring='f1',
    n_jobs=-1,
    return_train_score=True
)
dt_grid.fit(X_train, y_train)

print("=" * 60)
print("RESULTADOS - BÚSQUEDA DE HIPERPARÁMETROS (DECISION TREE)")
print("=" * 60)
print(f"Mejores hiperparámetros: {dt_grid.best_params_}")
print(f"Mejor F1 en validación cruzada: {dt_grid.best_score_:.4f}")

# %% [markdown]
# ### 6.3 Entrenamiento del modelo óptimo

# %%
# Entrenar con los mejores hiperparámetros
dt_best = dt_grid.best_estimator_

# Predicciones
y_train_pred_dt = dt_best.predict(X_train)
y_test_pred_dt = dt_best.predict(X_test)
y_test_proba_dt = dt_best.predict_proba(X_test)[:, 1]

# Métricas en train
print("=" * 60)
print("MÉTRICAS - DECISION TREE (TRAIN)")
print("=" * 60)
print(f"Accuracy:  {accuracy_score(y_train, y_train_pred_dt):.4f}")
print(f"F1-Score:  {f1_score(y_train, y_train_pred_dt):.4f}")

# Métricas en test
print("\n" + "=" * 60)
print("MÉTRICAS - DECISION TREE (TEST)")
print("=" * 60)
print(f"Accuracy:  {accuracy_score(y_test, y_test_pred_dt):.4f}")
print(f"Precision: {precision_score(y_test, y_test_pred_dt):.4f}")
print(f"Recall:    {recall_score(y_test, y_test_pred_dt):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_test_pred_dt):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_test_proba_dt):.4f}")

print("\n--- Classification Report (Test) ---")
print(classification_report(y_test, y_test_pred_dt, target_names=['Sin tensión', 'Tensión']))

# %% [markdown]
# ### 6.4 Matriz de confusión

# %%
# Matriz de confusión - Decision Tree
cm_dt = confusion_matrix(y_test, y_test_pred_dt)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Sin tensión', 'Tensión'],
            yticklabels=['Sin tensión', 'Tensión'],
            annot_kws={'size': 16})
ax.set_title('Matriz de Confusión - Árbol de Decisión (Test)', fontsize=13, fontweight='bold')
ax.set_ylabel('Valor Real', fontsize=12)
ax.set_xlabel('Valor Predicho', fontsize=12)
plt.tight_layout()
plt.savefig('images/05_confusion_matrix_decision_tree.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figura guardada: images/05_confusion_matrix_decision_tree.png")

print(f"Verdaderos Negativos (TN): {cm_dt[0,0]}")
print(f"Falsos Positivos (FP):     {cm_dt[0,1]}")
print(f"Falsos Negativos (FN):     {cm_dt[1,0]}")
print(f"Verdaderos Positivos (TP): {cm_dt[1,1]}")

# %% [markdown]
# ### 6.5 Visualización del árbol

# %%
# Representación textual del árbol
print("=" * 60)
print("ESTRUCTURA DEL ÁRBOL DE DECISIÓN")
print("=" * 60)
print(export_text(dt_best, feature_names=feature_cols, max_depth=5))

# %% [markdown]
# ### 6.6 Importancia de variables (Decision Tree)

# %%
# Importancia de variables - Decision Tree
dt_importance = pd.Series(dt_best.feature_importances_, index=feature_cols)
dt_importance = dt_importance.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
dt_importance.plot(kind='barh', ax=ax, color='#e67e22', edgecolor='black')
ax.set_title('Importancia de Variables - Árbol de Decisión', fontsize=14, fontweight='bold')
ax.set_xlabel('Importancia (Gini)')
plt.tight_layout()
plt.savefig('images/06_importancia_variables_decision_tree.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figura guardada: images/06_importancia_variables_decision_tree.png")

print("Variables utilizadas por el árbol (importancia > 0):")
for feat, imp in dt_importance[dt_importance > 0].sort_values(ascending=False).items():
    print(f"  {feat}: {imp:.4f}")

# %% [markdown]
# ### 6.7 Análisis de overfitting
# 
# La brecha entre el rendimiento en train y test es un indicador clave de sobreajuste. En este caso:
# - **Train Accuracy** es significativamente superior a **Test Accuracy**, lo que confirma cierto grado de overfitting.
# - Esto es esperado en árboles de decisión con datasets pequeños (60 muestras de entrenamiento).
# - Los hiperparámetros de regularización (`max_depth`, `min_samples_leaf`) mitigan parcialmente este problema.
# - La inestabilidad inherente del modelo (sensibilidad a la composición del dataset) contribuye a la brecha.

# %% [markdown]
# ## 7. Modelo 2: Random Forest
# 
# ### 7.1 **¿Por qué Random Forest es adecuado para FIRE-UdeA?**
# - Las interacciones entre variables financieras (ej: CFO bajo + endeudamiento alto + tendencia negativa) son mejor capturadas por múltiples árboles que por uno solo.
# - Con solo 80 observaciones, la regularización implícita del bagging ayuda a generalizar mejor.

# %% [markdown]
# ### 7.2 Búsqueda de hiperparámetros (GridSearchCV)

# %%
# ==============================
# 7. RANDOM FOREST
# ==============================

rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 3, 5],
    'max_features': ['sqrt', 'log2']
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params,
    cv=cv_strategy,
    scoring='f1',
    n_jobs=-1,
    return_train_score=True
)
rf_grid.fit(X_train, y_train)

print("=" * 60)
print("RESULTADOS - BÚSQUEDA DE HIPERPARÁMETROS (RANDOM FOREST)")
print("=" * 60)
print(f"Mejores hiperparámetros: {rf_grid.best_params_}")
print(f"Mejor F1 en validación cruzada: {rf_grid.best_score_:.4f}")

# %% [markdown]
# ### 7.3 Entrenamiento y evaluación

# %%
# Entrenar con los mejores hiperparámetros
rf_best = rf_grid.best_estimator_

# Predicciones
y_train_pred_rf = rf_best.predict(X_train)
y_test_pred_rf = rf_best.predict(X_test)
y_test_proba_rf = rf_best.predict_proba(X_test)[:, 1]

# Métricas en train
print("=" * 60)
print("MÉTRICAS - RANDOM FOREST (TRAIN)")
print("=" * 60)
print(f"Accuracy:  {accuracy_score(y_train, y_train_pred_rf):.4f}")
print(f"F1-Score:  {f1_score(y_train, y_train_pred_rf):.4f}")

# Métricas en test
print("\n" + "=" * 60)
print("MÉTRICAS - RANDOM FOREST (TEST)")
print("=" * 60)
print(f"Accuracy:  {accuracy_score(y_test, y_test_pred_rf):.4f}")
print(f"Precision: {precision_score(y_test, y_test_pred_rf):.4f}")
print(f"Recall:    {recall_score(y_test, y_test_pred_rf):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_test_pred_rf):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_test_proba_rf):.4f}")

print("\n--- Classification Report (Test) ---")
print(classification_report(y_test, y_test_pred_rf, target_names=['Sin tensión', 'Tensión']))

# %% [markdown]
# ### 7.4 Matriz de confusión

# %%
# Matriz de confusión - Random Forest
cm_rf = confusion_matrix(y_test, y_test_pred_rf)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=ax,
            xticklabels=['Sin tensión', 'Tensión'],
            yticklabels=['Sin tensión', 'Tensión'],
            annot_kws={'size': 16})
ax.set_title('Matriz de Confusión - Random Forest (Test)', fontsize=13, fontweight='bold')
ax.set_ylabel('Valor Real', fontsize=12)
ax.set_xlabel('Valor Predicho', fontsize=12)
plt.tight_layout()
plt.savefig('images/07_confusion_matrix_random_forest.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figura guardada: images/07_confusion_matrix_random_forest.png")

print(f"Verdaderos Negativos (TN): {cm_rf[0,0]}")
print(f"Falsos Positivos (FP):     {cm_rf[0,1]}")
print(f"Falsos Negativos (FN):     {cm_rf[1,0]}")
print(f"Verdaderos Positivos (TP): {cm_rf[1,1]}")

# %% [markdown]
# ### 7.5 Importancia de variables (Random Forest)

# %%
# Importancia de variables - Random Forest
rf_importance = pd.Series(rf_best.feature_importances_, index=feature_cols)
rf_importance = rf_importance.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
rf_importance.plot(kind='barh', ax=ax, color='#2ecc71', edgecolor='black')
ax.set_title('Importancia de Variables - Random Forest', fontsize=14, fontweight='bold')
ax.set_xlabel('Importancia (Mean Decrease in Impurity)')
plt.tight_layout()
plt.savefig('images/08_importancia_variables_random_forest.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figura guardada: images/08_importancia_variables_random_forest.png")

print("\nRanking de importancia de variables (Random Forest):")
print("=" * 50)
for rank, (feat, imp) in enumerate(rf_importance.sort_values(ascending=False).items(), 1):
    print(f"  {rank:2d}. {feat:30s} {imp:.4f}")

# %% [markdown]
# ## 8. Modelo de Referencia: Gradient Boosting

# %%
# ==============================
# 8. GRADIENT BOOSTING (referencia)
# ==============================
gb = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)
gb.fit(X_train, y_train)

y_train_pred_gb = gb.predict(X_train)
y_test_pred_gb = gb.predict(X_test)
y_test_proba_gb = gb.predict_proba(X_test)[:, 1]

print("=" * 60)
print("MÉTRICAS - GRADIENT BOOSTING (REFERENCIA)")
print("=" * 60)
print(f"Train Accuracy: {accuracy_score(y_train, y_train_pred_gb):.4f}")
print(f"\nTest Accuracy:  {accuracy_score(y_test, y_test_pred_gb):.4f}")
print(f"Test Precision: {precision_score(y_test, y_test_pred_gb):.4f}")
print(f"Test Recall:    {recall_score(y_test, y_test_pred_gb):.4f}")
print(f"Test F1-Score:  {f1_score(y_test, y_test_pred_gb):.4f}")
print(f"Test ROC-AUC:   {roc_auc_score(y_test, y_test_proba_gb):.4f}")

print("\n--- Classification Report (Test) ---")
print(classification_report(y_test, y_test_pred_gb, target_names=['Sin tensión', 'Tensión']))

# %% [markdown]
# ## 9. Evaluación Comparativa de Modelos

# %% [markdown]
# ### 9.1 Tabla comparativa de métricas

# %%
# ==============================
# 9. COMPARACIÓN DE MODELOS
# ==============================

models = {
    'Decision Tree': (dt_best, y_test_pred_dt, y_test_proba_dt, y_train_pred_dt),
    'Random Forest': (rf_best, y_test_pred_rf, y_test_proba_rf, y_train_pred_rf),
    'Gradient Boosting': (gb, y_test_pred_gb, y_test_proba_gb, y_train_pred_gb)
}

results = []
for name, (mdl, y_pred, y_proba, y_pred_tr) in models.items():
    results.append({
        'Modelo': name,
        'Train Acc': accuracy_score(y_train, y_pred_tr),
        'Test Acc': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba)
    })

df_results = pd.DataFrame(results).set_index('Modelo')
print("=" * 80)
print("TABLA COMPARATIVA DE MÉTRICAS")
print("=" * 80)
print(df_results.round(4).to_string())

# %% [markdown]
# ### 9.2 Matrices de confusión comparadas

# %%
# Matrices de confusión comparadas
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
cmaps = ['Oranges', 'Greens', 'Blues']
for i, (name, (mdl, y_pred, _, _)) in enumerate(models.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmaps[i], ax=axes[i],
                xticklabels=['Sin tensión', 'Tensión'],
                yticklabels=['Sin tensión', 'Tensión'],
                annot_kws={'size': 14})
    axes[i].set_title(f'{name}', fontsize=12, fontweight='bold')
    axes[i].set_ylabel('Real')
    axes[i].set_xlabel('Predicho')

plt.suptitle('Matrices de Confusión - Comparación (Test)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('images/09_confusion_matrices_comparacion.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figura guardada: images/09_confusion_matrices_comparacion.png")

# %% [markdown]
# ### 9.3 Gráfico comparativo de métricas

# %%
# Gráfico comparativo de barras
metrics_plot = df_results[['Test Acc', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']]

fig, ax = plt.subplots(figsize=(10, 6))
metrics_plot.plot(kind='bar', ax=ax, color=['#34495e', '#e74c3c', '#f39c12', '#2ecc71', '#3498db'],
                  edgecolor='black')
ax.set_title('Comparación de Métricas por Modelo (Test)', fontsize=14, fontweight='bold')
ax.set_ylabel('Score')
ax.set_ylim(0, 1)
ax.legend(loc='upper left', fontsize=9)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('images/10_comparacion_metricas.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figura guardada: images/10_comparacion_metricas.png")

# %% [markdown]
# ### 9.4 Validación cruzada

# %%
# Validación cruzada (5-fold, estratificada)
print("=" * 60)
print("VALIDACIÓN CRUZADA (5-Fold Estratificada)")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, mdl in [('Decision Tree', dt_best), ('Random Forest', rf_best), ('Gradient Boosting', gb)]:
    f1_scores = cross_val_score(mdl, X, y, cv=cv, scoring='f1')
    acc_scores = cross_val_score(mdl, X, y, cv=cv, scoring='accuracy')
    print(f"\n{name}:")
    print(f"  F1  → media: {f1_scores.mean():.4f} ± {f1_scores.std():.4f}  |  folds: {[f'{s:.3f}' for s in f1_scores]}")
    print(f"  Acc → media: {acc_scores.mean():.4f} ± {acc_scores.std():.4f}  |  folds: {[f'{s:.3f}' for s in acc_scores]}")

# %% [markdown]
# ## 10. Interpretación Financiera
# 
# ### 10.1 Variables más importantes según Random Forest
# 
# Los resultados del Random Forest (modelo con mejor desempeño) revelan las siguientes variables como las más relevantes para predecir tensión financiera:
# 
# **1. CFO (flujo de caja operativo) — Importancia: ~17.6%**
# - El flujo de caja operativo es el indicador más determinante. Un CFO negativo o decreciente indica que la universidad no genera suficiente efectivo con sus operaciones para cubrir sus obligaciones corrientes.
# - **Lectura institucional:** La UdeA debe monitorear de cerca la generación operativa de caja. Periodos prolongados con CFO negativo señalan problemas estructurales, no solo coyunturales.
# 
# **2. GP_RATIO (gastos de personal / ingresos) — Importancia: ~14.4%**
# - Un ratio alto de gastos de personal indica rigidez presupuestal. Cuando la nómina consume una proporción excesiva de los ingresos, la universidad pierde capacidad de maniobra financiera.
# - **Lectura institucional:** La presión del gasto de personal es uno de los principales factores de riesgo. Cualquier estrategia de sostenibilidad debe considerar la contención o eficiencia de este componente.
# 
# **3. Días de efectivo — Importancia: ~13.3%**
# - Mide cuántos días puede operar la universidad con el efectivo disponible. Valores bajos indican vulnerabilidad ante demoras en transferencias o ingresos estacionales.
# - **Lectura institucional:** Una universidad pública con menos de 30 días de efectivo enfrenta riesgo de incumplir nómina o proveedores.
# 
# **4. Endeudamiento — Importancia: ~12.3%**
# - Un endeudamiento creciente combinado con baja liquidez potencia el riesgo. El modelo identifica esta variable como un amplificador de vulnerabilidad.
# 
# **5. Tendencia de ingresos, liquidez, HHI — Importancias: 5-8%**
# - La tendencia negativa de ingresos anticipa deterioro futuro.
# - La concentración de fuentes (HHI alto) indica dependencia excesiva de pocas fuentes, aumentando la vulnerabilidad ante cambios regulatorios o presupuestales.
# 
# ### 10.2 Implicaciones para la Universidad de Antioquia
# 
# Los hallazgos del modelo confirman que la sostenibilidad financiera de la UdeA depende de un equilibrio entre:
# - **Generación de caja operativa** suficiente para cubrir obligaciones.
# - **Contención del gasto de personal** en relación con los ingresos.
# - **Diversificación de fuentes de ingreso** para reducir dependencia de la Ley 30.
# - **Mantenimiento de reservas de liquidez** (días de efectivo) como buffer ante contingencias.
# - **Gestión prudente del endeudamiento** para no amplificar la vulnerabilidad.

# %% [markdown]
# ## 11. Conclusiones
# 
# ### 11.1 Conclusiones técnicas
# 
# **Modelo ganador: Random Forest**
# 
# | Aspecto | Decision Tree | Random Forest | Gradient Boosting |
# |---------|:---:|:---:|:---:|
# | Test Accuracy | 0.60 | **0.75** | 0.65 |
# | F1-Score | 0.60 | **0.74** | 0.59 |
# | ROC-AUC | 0.65 | **0.87** | 0.68 |
# | Riesgo de overfitting | Alto | Moderado | Alto |
# | Interpretabilidad | Alta | Media | Baja |
# 
# - **Random Forest superó tanto al árbol de decisión individual como al Gradient Boosting**, con un ROC-AUC de 0.87 frente a 0.65 y 0.68 respectivamente.
# - El resultado es notable considerando el tamaño reducido del dataset (80 observaciones, 60 para entrenamiento).
# - La validación cruzada confirma la superioridad del Random Forest, con un F1 medio de ~0.68 y menor variabilidad que el árbol individual.
# - El Gradient Boosting, a pesar de ser un modelo potente, no logra superar al Random Forest probablemente debido al tamaño reducido del dataset, donde el boosting tiende a sobreajustar más que el bagging.
# 
# ### 11.2 Conclusiones financieras e institucionales
# 
# - Los **factores que mejor explican el riesgo financiero** de la UdeA son, en orden: flujo de caja operativo, ratio de gastos de personal, días de efectivo disponible, endeudamiento y tendencia de ingresos.
# - El modelo puede servir como **herramienta de monitoreo y alerta temprana**: cada trimestre o semestre, al alimentar los indicadores actualizados, el modelo puede señalar qué unidades están en riesgo.
# - La **dependencia de la Ley 30** y la **rigidez del gasto de personal** emergen como factores estructurales que deben abordarse con políticas de largo plazo.
# - Se recomienda a la administración universitaria mantener un dashboard de indicadores financieros con los umbrales críticos identificados por el modelo (ej: CFO < 0 sostenido, gp_ratio > 0.55, días de efectivo < 60).