# %% [markdown]
# # Random Forest - Semana 7
# ## SI3015 - Fundamentos de Aprendizaje Automático
#
# **Autor:** Alejandro Garcés Ramírez
# **Fecha:** Marzo 2026
# **Dataset:** Breast Cancer Wisconsin (sklearn)
#
# ---
#
# **Objetivo:** Implementar Random Forest como método de ensamble basado en árboles de decisión,
# comprender el concepto de bagging, comparar su desempeño contra un árbol individual,
# y analizar la importancia de características en un problema de clasificación binaria.

# %% [markdown]
# ## 1. Comprensión del Problema
#
# ### 1.1 ¿Qué es Random Forest?
#
# Random Forest es un método de **ensamble** que construye múltiples árboles de decisión
# durante el entrenamiento y combina sus predicciones mediante votación (clasificación) o
# promedio (regresión). Su fortaleza radica en dos mecanismos:
#
# - **Bagging (Bootstrap Aggregating):** Cada árbol se entrena con una muestra aleatoria
#   con reemplazo del conjunto de entrenamiento → reduce varianza.
# - **Feature Randomness:** En cada split se evalúa solo un subconjunto aleatorio de
#   características (`max_features`) → decorrelaciona los árboles.
#
# ### 1.2 Árbol Individual vs Random Forest
#
# | Característica     | Árbol Individual      | Random Forest            |
# |--------------------|-----------------------|--------------------------|
# | Varianza           | Alta (overfitting)    | Baja (promedio de árboles) |
# | Sesgo              | Bajo                  | Bajo                     |
# | Interpretabilidad  | Alta                  | Media (importancias)     |
# | Velocidad          | Rápido                | Más lento                |
# | Robustez           | Sensible a ruido      | Robusto                  |
#
# ### 1.3 Dataset: Breast Cancer Wisconsin
#
# - **569 muestras:** tumores mamarios con características extraídas de imágenes digitalizadas
# - **30 características:** medidas del núcleo celular (radio, textura, perímetro, área, etc.)
# - **2 clases:** Maligno (0) vs Benigno (1) → clasificación binaria
# - **Aplicación:** Diagnóstico médico asistido por ML
#
# ### 1.4 Error Out-of-Bag (OOB)
#
# Como cada árbol se entrena con ~63% de los datos (bootstrap), el 37% restante
# ("out-of-bag") puede usarse como conjunto de validación interno sin necesidad
# de un split adicional. El OOB error es una estimación gratuita del error de generalización.

# %% [markdown]
# ## 2. Carga de Datos

# %%
# ==============================
# 1. IMPORTS Y CONFIGURACIÓN
# ==============================
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     GridSearchCV, StratifiedKFold)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score,
                             roc_curve, precision_score, recall_score)
import warnings
warnings.filterwarnings('ignore')

os.makedirs('outputs', exist_ok=True)

plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
sns.set_style('whitegrid')

print("Librerías cargadas correctamente.")

# %%
# Cargar dataset Breast Cancer
bc = load_breast_cancer()
X = bc.data
y = bc.target
feature_names = bc.feature_names
class_names = bc.target_names  # ['malignant', 'benign']

df = pd.DataFrame(X, columns=feature_names)
df['diagnosis'] = [class_names[i] for i in y]

print(f"Dataset Breast Cancer cargado: {X.shape[0]} muestras x {X.shape[1]} características")
print(f"Clases: {list(class_names)}")
print(f"\nDistribución de clases:")
print(df['diagnosis'].value_counts())
print(f"\nPrimeras filas (primeras 6 columnas):")
print(df.iloc[:5, :6])

# %% [markdown]
# ## 3. Análisis Exploratorio de Datos (EDA)

# %%
# ==============================
# 2. EDA
# ==============================
print("=" * 60)
print("ESTADÍSTICAS DESCRIPTIVAS (primeras 5 features)")
print("=" * 60)
print(df.iloc[:, :5].describe())
print(f"\nValores faltantes: {df.isnull().sum().sum()}")

# %%
# Visualización 1: Distribución de clases + boxplots de features clave
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Distribución de clases
class_counts = df['diagnosis'].value_counts()
colors_pie = ['#E53935', '#43A047']
axes[0].pie(class_counts, labels=class_counts.index, autopct='%1.1f%%',
            colors=colors_pie, startangle=90)
axes[0].set_title('Distribución de Clases')

# Boxplot: mean radius por clase
df.boxplot(column='mean radius', by='diagnosis', ax=axes[1],
           patch_artist=True)
axes[1].set_title('Radio Medio por Diagnóstico')
axes[1].set_xlabel('Diagnóstico')
axes[1].set_ylabel('mean radius')

# Boxplot: mean texture por clase
df.boxplot(column='mean texture', by='diagnosis', ax=axes[2],
           patch_artist=True)
axes[2].set_title('Textura Media por Diagnóstico')
axes[2].set_xlabel('Diagnóstico')
axes[2].set_ylabel('mean texture')

plt.suptitle('')
plt.tight_layout()
plt.savefig('outputs/01_eda_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("Guardado: outputs/01_eda_overview.png")

# %%
# Visualización 2: Distribuciones de las 6 features más relevantes
top_features = ['mean radius', 'mean texture', 'mean perimeter',
                'mean area', 'mean smoothness', 'mean concavity']

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for i, feat in enumerate(top_features):
    for cls, color in zip(class_names, ['#E53935', '#43A047']):
        mask = df['diagnosis'] == cls
        axes[i].hist(df.loc[mask, feat], bins=20, alpha=0.6,
                     label=cls, color=color)
    axes[i].set_title(feat)
    axes[i].set_xlabel(feat)
    axes[i].set_ylabel('Frecuencia')
    axes[i].legend(fontsize=8)

plt.suptitle('Distribuciones de Características Clave por Diagnóstico', fontsize=13)
plt.tight_layout()
plt.savefig('outputs/02_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Guardado: outputs/02_feature_distributions.png")

# %%
# Visualización 3: Correlación (primeras 10 features)
fig, ax = plt.subplots(figsize=(10, 8))
corr = df.drop('diagnosis', axis=1).iloc[:, :10].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax, linewidths=0.4, annot_kws={'size': 8})
ax.set_title('Matriz de Correlación (primeras 10 características)')
plt.tight_layout()
plt.savefig('outputs/03_correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Guardado: outputs/03_correlation_matrix.png")

# %% [markdown]
# ## 4. División de Datos (Train / Test)

# %%
# ==============================
# 3. TRAIN / TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Entrenamiento: {X_train.shape[0]} muestras")
print(f"Prueba:        {X_test.shape[0]} muestras")
print(f"Distribución clases (train): Maligno={np.sum(y_train==0)}, Benigno={np.sum(y_train==1)}")
print(f"Distribución clases (test):  Maligno={np.sum(y_test==0)}, Benigno={np.sum(y_test==1)}")

# %% [markdown]
# ## 5. Árbol de Decisión Individual (Baseline)

# %%
# ==============================
# 4. ÁRBOL DE DECISIÓN (BASELINE)
# ==============================
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

acc_dt = accuracy_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt, average='weighted')
prec_dt = precision_score(y_test, y_pred_dt, average='weighted')
rec_dt = recall_score(y_test, y_pred_dt, average='weighted')
auc_dt = roc_auc_score(y_test, dt.predict_proba(X_test)[:, 1])

print("Árbol de Decisión (max_depth=5):")
print(f"  Accuracy:  {acc_dt:.4f}")
print(f"  F1 Score:  {f1_dt:.4f}")
print(f"  Precision: {prec_dt:.4f}")
print(f"  Recall:    {rec_dt:.4f}")
print(f"  ROC-AUC:   {auc_dt:.4f}")

# %% [markdown]
# ## 6. Búsqueda de Hiperparámetros para Random Forest

# %%
# ==============================
# 5. GRIDSEARCHCV - RANDOM FOREST
# ==============================
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'max_features': ['sqrt', 'log2'],
    'min_samples_leaf': [1, 2, 4]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search_rf = GridSearchCV(
    RandomForestClassifier(random_state=42, oob_score=True),
    param_grid=param_grid_rf,
    cv=cv,
    scoring='f1_weighted',
    n_jobs=-1
)
grid_search_rf.fit(X_train, y_train)

print("Mejores hiperparámetros (Random Forest):")
for param, value in grid_search_rf.best_params_.items():
    print(f"  {param}: {value}")
print(f"  CV F1 Score (mejor): {grid_search_rf.best_score_:.4f}")

# %% [markdown]
# ## 7. Random Forest Optimizado

# %%
# ==============================
# 6. RANDOM FOREST OPTIMIZADO
# ==============================
rf_best = grid_search_rf.best_estimator_
y_pred_rf = rf_best.predict(X_test)

acc_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')
prec_rf = precision_score(y_test, y_pred_rf, average='weighted')
rec_rf = recall_score(y_test, y_pred_rf, average='weighted')
auc_rf = roc_auc_score(y_test, rf_best.predict_proba(X_test)[:, 1])

print(f"\nRandom Forest optimizado:")
print(f"  Accuracy:      {acc_rf:.4f}")
print(f"  F1 Score:      {f1_rf:.4f}")
print(f"  Precision:     {prec_rf:.4f}")
print(f"  Recall:        {rec_rf:.4f}")
print(f"  ROC-AUC:       {auc_rf:.4f}")
print(f"  OOB Score:     {rf_best.oob_score_:.4f}")
print(f"  N° de árboles: {rf_best.n_estimators}")

print("\nReporte de clasificación (Random Forest):")
print(classification_report(y_test, y_pred_rf, target_names=class_names))

# Cross-validation
cv_scores_rf = cross_val_score(rf_best, X, y, cv=cv, scoring='f1_weighted')
print(f"Cross-Validation (5-fold) F1: {cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.4f}")

# %% [markdown]
# ## 8. Evaluación y Comparación

# %%
# ==============================
# 7. EVALUACIÓN
# ==============================

# Matrices de confusión: Árbol vs Random Forest
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, y_pred, title, acc in zip(
    axes,
    [y_pred_dt, y_pred_rf],
    ['Árbol de Decisión', 'Random Forest'],
    [acc_dt, acc_rf]
):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Real')
    ax.set_title(f'{title}\nAccuracy: {acc:.4f}')

plt.suptitle('Matrices de Confusión - Árbol vs Random Forest', fontsize=13)
plt.tight_layout()
plt.savefig('outputs/04_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("Guardado: outputs/04_confusion_matrices.png")

# %%
# Curvas ROC
fig, ax = plt.subplots(figsize=(8, 6))

for model, y_pred, name, color in [
    (dt, y_pred_dt, 'Árbol de Decisión', '#FF6F00'),
    (rf_best, y_pred_rf, 'Random Forest', '#2196F3')
]:
    proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc_val = roc_auc_score(y_test, proba)
    ax.plot(fpr, tpr, label=f'{name} (AUC = {auc_val:.4f})', color=color, linewidth=2)

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Clasificador aleatorio')
ax.set_xlabel('Tasa de Falsos Positivos')
ax.set_ylabel('Tasa de Verdaderos Positivos')
ax.set_title('Curva ROC — Árbol de Decisión vs Random Forest')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/05_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Guardado: outputs/05_roc_curves.png")

# %%
# Importancia de características (Random Forest - Top 15)
importances_rf = rf_best.feature_importances_
indices_rf = np.argsort(importances_rf)[::-1][:15]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(range(15), importances_rf[indices_rf], color='#2196F3', alpha=0.85)
ax.set_xticks(range(15))
ax.set_xticklabels([feature_names[i] for i in indices_rf],
                   rotation=40, ha='right', fontsize=9)
ax.set_title('Top 15 Características Más Importantes - Random Forest')
ax.set_ylabel('Importancia (Mean Decrease in Impurity)')

for bar, imp in zip(bars, importances_rf[indices_rf]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
            f'{imp:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('outputs/06_feature_importance_rf.png', dpi=150, bbox_inches='tight')
plt.close()
print("Guardado: outputs/06_feature_importance_rf.png")

# %%
# Comparación Árbol vs Random Forest
fig, ax = plt.subplots(figsize=(9, 5))

metrics_labels = ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'ROC-AUC']
dt_vals = [acc_dt, f1_dt, prec_dt, rec_dt, auc_dt]
rf_vals = [acc_rf, f1_rf, prec_rf, rec_rf, auc_rf]

x = np.arange(len(metrics_labels))
width = 0.35

bars1 = ax.bar(x - width / 2, dt_vals, width, label='Árbol de Decisión', color='#FF6F00', alpha=0.85)
bars2 = ax.bar(x + width / 2, rf_vals, width, label='Random Forest', color='#2196F3', alpha=0.85)

for bar, val in zip(bars1, dt_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9)
for bar, val in zip(bars2, rf_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9)

ax.set_ylim(0.85, 1.05)
ax.set_xticks(x)
ax.set_xticklabels(metrics_labels)
ax.set_ylabel('Métrica')
ax.set_title('Comparación de Métricas: Árbol de Decisión vs Random Forest')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/07_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Guardado: outputs/07_model_comparison.png")

# %%
# Efecto del número de árboles sobre el accuracy
n_estimators_range = [10, 25, 50, 75, 100, 150, 200, 300]
acc_per_n = []

for n in n_estimators_range:
    rf_tmp = RandomForestClassifier(n_estimators=n, random_state=42,
                                    **{k: v for k, v in grid_search_rf.best_params_.items()
                                       if k != 'n_estimators'})
    rf_tmp.fit(X_train, y_train)
    acc_per_n.append(accuracy_score(y_test, rf_tmp.predict(X_test)))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(n_estimators_range, acc_per_n, marker='o', color='#2196F3', linewidth=2)
ax.set_xlabel('Número de Árboles (n_estimators)')
ax.set_ylabel('Accuracy (test)')
ax.set_title('Efecto del Número de Árboles sobre el Accuracy')
ax.grid(alpha=0.4)
ax.set_xticks(n_estimators_range)

plt.tight_layout()
plt.savefig('outputs/08_n_estimators_effect.png', dpi=150, bbox_inches='tight')
plt.close()
print("Guardado: outputs/08_n_estimators_effect.png")

# %% [markdown]
# ## 9. Resumen de Resultados

# %%
# ==============================
# 8. RESUMEN FINAL
# ==============================
print("\n" + "=" * 70)
print("RESUMEN COMPARATIVO - ÁRBOL DE DECISIÓN vs RANDOM FOREST")
print("=" * 70)
print(f"\n{'Métrica':<18} {'Árbol Decisión':>18} {'Random Forest':>16}")
print("-" * 55)
for name, dt_v, rf_v in zip(metrics_labels, dt_vals, rf_vals):
    diff = rf_v - dt_v
    sign = '+' if diff >= 0 else ''
    print(f"{name:<18} {dt_v:>18.4f} {rf_v:>16.4f}  ({sign}{diff:.4f})")

print(f"\n{'OOB Score (RF)':<18} {'—':>18} {rf_best.oob_score_:>16.4f}")
print(f"\nMejores hiperparámetros RF: {grid_search_rf.best_params_}")
print(f"CV F1 (5-fold): {cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.4f}")

top_feature_rf = feature_names[np.argmax(rf_best.feature_importances_)]
print(f"\nCaracterística más importante (RF): {top_feature_rf}")
print(f"  Importancia: {rf_best.feature_importances_.max():.4f}")

print("\nArchivos generados en outputs/:")
for f in sorted(os.listdir('outputs')):
    print(f"  {f}")
