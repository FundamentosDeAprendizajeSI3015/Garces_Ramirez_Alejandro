# %% [markdown]
# # Árboles de Decisión - Semana 6
# ## SI3015 - Fundamentos de Aprendizaje Automático
#
# **Autor:** Alejandro Garcés Ramírez
# **Fecha:** Marzo 2026
# **Dataset:** Iris Dataset (sklearn)
#
# ---
#
# **Objetivo:** Implementar y comprender los árboles de decisión como método de clasificación
# supervisada, explorando su interpretabilidad, el efecto del podado (pruning) y la
# selección de hiperparámetros mediante búsqueda en cuadrícula.

# %% [markdown]
# ## 1. Comprensión del Problema
#
# ### 1.1 ¿Qué es un Árbol de Decisión?
#
# Un árbol de decisión es un modelo de aprendizaje supervisado que divide el espacio de
# características en regiones mediante preguntas binarias sucesivas. En cada nodo interno
# se evalúa una condición (ej: petal_length <= 2.45), y la predicción se asigna en las hojas.
#
# ### 1.2 ¿Por qué el dataset Iris?
#
# El dataset Iris es ideal para introducir árboles de decisión porque:
# - **Interpretabilidad:** 4 características con separación clara entre clases
# - **Multiclase:** 3 especies → permite visualizar splits múltiples
# - **Tamaño manejable:** 150 instancias, sin necesidad de limpieza compleja
# - **Bien estudiado:** Los resultados son verificables y esperados
#
# ### 1.3 Criterios de División
#
# Los árboles usan métricas de impureza para decidir cada split:
# - **Gini:** `Gini = 1 - Σ(pᵢ²)` → probabilidad de clasificación incorrecta
# - **Entropía:** `H = -Σ(pᵢ * log₂(pᵢ))` → medida de desorden informacional
#
# El algoritmo CART elige el split que maximiza la Ganancia de Información.
#
# ### 1.4 Problema del Sobreajuste (Overfitting)
#
# Un árbol sin restricciones memorizará el conjunto de entrenamiento.
# El **podado (pruning)** controla la complejidad limitando:
# - `max_depth`: profundidad máxima del árbol
# - `min_samples_split`: mínimo de muestras para dividir un nodo
# - `min_samples_leaf`: mínimo de muestras en una hoja

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

from sklearn.datasets import load_iris
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     GridSearchCV, StratifiedKFold)
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
import warnings
warnings.filterwarnings('ignore')

os.makedirs('outputs', exist_ok=True)

plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
sns.set_style('whitegrid')

print("Librerías cargadas correctamente.")

# %%
# Cargar dataset Iris
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
class_names = iris.target_names

df = pd.DataFrame(X, columns=feature_names)
df['species'] = [class_names[i] for i in y]

print(f"Dataset Iris cargado: {X.shape[0]} muestras x {X.shape[1]} características")
print(f"Clases: {list(class_names)}")
print(f"\nPrimeras filas:")
print(df.head(10))

# %% [markdown]
# ## 3. Análisis Exploratorio de Datos (EDA)

# %%
# ==============================
# 2. EDA
# ==============================
print("=" * 60)
print("ESTADÍSTICAS DESCRIPTIVAS")
print("=" * 60)
print(df.describe())

print(f"\nDistribución de clases:")
print(df['species'].value_counts())
print(f"\nValores faltantes: {df.isnull().sum().sum()}")

# %%
# Visualización 1: Distribución de clases + scatter
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

species_counts = df['species'].value_counts()
axes[0].pie(species_counts, labels=species_counts.index, autopct='%1.1f%%',
            colors=['#2196F3', '#4CAF50', '#FF9800'], startangle=90)
axes[0].set_title('Distribución de Clases')

color_map = {'setosa': '#2196F3', 'versicolor': '#4CAF50', 'virginica': '#FF9800'}
for species, color in color_map.items():
    mask = df['species'] == species
    axes[1].scatter(df.loc[mask, 'petal length (cm)'],
                    df.loc[mask, 'petal width (cm)'],
                    label=species, color=color, alpha=0.7, edgecolors='k', linewidths=0.3)
axes[1].set_xlabel('Longitud del Pétalo (cm)')
axes[1].set_ylabel('Ancho del Pétalo (cm)')
axes[1].set_title('Dispersión: Longitud vs Ancho del Pétalo')
axes[1].legend()

plt.suptitle('EDA - Dataset Iris', fontsize=14)
plt.tight_layout()
plt.savefig('outputs/01_eda_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("Guardado: outputs/01_eda_overview.png")

# %%
# Visualización 2: Distribuciones por característica
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
palette = {'setosa': '#2196F3', 'versicolor': '#4CAF50', 'virginica': '#FF9800'}

for i, feature in enumerate(feature_names):
    for species in class_names:
        mask = df['species'] == species
        axes[i].hist(df.loc[mask, feature], bins=15,
                     alpha=0.6, label=species, color=palette[species])
    axes[i].set_title(f'Distribución de {feature}')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Frecuencia')
    axes[i].legend()

plt.suptitle('Distribuciones de Características por Especie', fontsize=14)
plt.tight_layout()
plt.savefig('outputs/02_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Guardado: outputs/02_feature_distributions.png")

# %%
# Visualización 3: Matriz de correlación
fig, ax = plt.subplots(figsize=(7, 5))
corr = df.drop('species', axis=1).corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax, square=True, linewidths=0.5)
ax.set_title('Matriz de Correlación - Iris Dataset')
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
print(f"Distribución clases (train): {np.bincount(y_train)}")
print(f"Distribución clases (test):  {np.bincount(y_test)}")

# %% [markdown]
# ## 5. Árbol de Decisión Sin Podado

# %%
# ==============================
# 4. ÁRBOL SIN PODADO
# ==============================
dt_full = DecisionTreeClassifier(random_state=42)
dt_full.fit(X_train, y_train)

y_pred_full = dt_full.predict(X_test)
acc_full = accuracy_score(y_test, y_pred_full)
f1_full = f1_score(y_test, y_pred_full, average='weighted')

print(f"Árbol completo (sin podado):")
print(f"  Profundidad:  {dt_full.get_depth()}")
print(f"  N° de hojas:  {dt_full.get_n_leaves()}")
print(f"  Accuracy:     {acc_full:.4f}")
print(f"  F1 Score:     {f1_full:.4f}")

# Visualizar árbol completo
fig, ax = plt.subplots(figsize=(22, 10))
plot_tree(dt_full, feature_names=feature_names, class_names=class_names,
          filled=True, rounded=True, ax=ax, fontsize=8)
plt.title('Árbol de Decisión - Sin Podado (Árbol Completo)', fontsize=13)
plt.tight_layout()
plt.savefig('outputs/04_tree_full.png', dpi=120, bbox_inches='tight')
plt.close()
print("Guardado: outputs/04_tree_full.png")

# Imprimir reglas textuales (primeros 3 niveles)
print("\nReglas del árbol (depth <= 3):")
rules = export_text(dt_full, feature_names=list(feature_names), max_depth=3)
print(rules)

# %% [markdown]
# ## 6. Búsqueda de Hiperparámetros (GridSearchCV)

# %%
# ==============================
# 5. GRIDSEARCHCV
# ==============================
param_grid = {
    'max_depth': [2, 3, 4, 5, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=cv,
    scoring='f1_weighted',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print("Mejores hiperparámetros encontrados:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"  CV F1 Score (mejor): {grid_search.best_score_:.4f}")

# %% [markdown]
# ## 7. Árbol de Decisión Optimizado (con Podado)

# %%
# ==============================
# 6. ÁRBOL OPTIMIZADO
# ==============================
dt_best = grid_search.best_estimator_
y_pred_best = dt_best.predict(X_test)
acc_best = accuracy_score(y_test, y_pred_best)
f1_best = f1_score(y_test, y_pred_best, average='weighted')

print(f"\nÁrbol optimizado (con podado):")
print(f"  Profundidad:  {dt_best.get_depth()}")
print(f"  N° de hojas:  {dt_best.get_n_leaves()}")
print(f"  Accuracy:     {acc_best:.4f}")
print(f"  F1 Score:     {f1_best:.4f}")

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred_best, target_names=class_names))

# Cross-validation final
cv_scores = cross_val_score(dt_best, X, y, cv=cv, scoring='f1_weighted')
print(f"Cross-Validation (5-fold) F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Visualizar árbol optimizado
fig, ax = plt.subplots(figsize=(14, 8))
plot_tree(dt_best, feature_names=feature_names, class_names=class_names,
          filled=True, rounded=True, ax=ax, fontsize=10)
plt.title(f'Árbol de Decisión Optimizado — depth={dt_best.get_depth()}, '
          f'criterion={dt_best.criterion}', fontsize=13)
plt.tight_layout()
plt.savefig('outputs/05_tree_pruned.png', dpi=150, bbox_inches='tight')
plt.close()
print("Guardado: outputs/05_tree_pruned.png")

# %% [markdown]
# ## 8. Evaluación del Modelo

# %%
# ==============================
# 7. EVALUACIÓN
# ==============================

# Matrices de confusión: sin podado vs optimizado
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, y_pred, title in zip(axes,
                              [y_pred_full, y_pred_best],
                              ['Sin Podado', 'Optimizado']):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Real')
    acc = accuracy_score(y_test, y_pred)
    ax.set_title(f'Árbol {title}\nAccuracy: {acc:.4f}')

plt.suptitle('Matrices de Confusión - Árbol de Decisión', fontsize=14)
plt.tight_layout()
plt.savefig('outputs/06_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("Guardado: outputs/06_confusion_matrices.png")

# %%
# Importancia de características
importances = dt_best.feature_importances_
indices = np.argsort(importances)[::-1]
colors_bar = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(range(len(feature_names)),
              importances[indices],
              color=[colors_bar[i] for i in indices])
ax.set_xticks(range(len(feature_names)))
ax.set_xticklabels([feature_names[i] for i in indices], rotation=20, ha='right')
ax.set_title('Importancia de Características - Árbol de Decisión Optimizado')
ax.set_ylabel('Importancia (Reducción de Impureza Gini)')

for bar, imp in zip(bars, importances[indices]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f'{imp:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('outputs/07_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Guardado: outputs/07_feature_importance.png")

# %%
# Comparación: árbol completo vs optimizado
fig, ax = plt.subplots(figsize=(8, 5))

labels = ['Sin Podado', 'Optimizado']
acc_vals = [acc_full, acc_best]
f1_vals = [f1_full, f1_best]

x = np.arange(len(labels))
width = 0.3

bars1 = ax.bar(x - width / 2, acc_vals, width, label='Accuracy', color='#2196F3')
bars2 = ax.bar(x + width / 2, f1_vals, width, label='F1 Score', color='#4CAF50')

for bar, val in zip(bars1, acc_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10)
for bar, val in zip(bars2, f1_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10)

depth_labels = [f'Prof.={dt_full.get_depth()}', f'Prof.={dt_best.get_depth()}']
for i, label in enumerate(depth_labels):
    ax.text(i, 0.82, label, ha='center', va='bottom',
            color='white', fontsize=9, fontweight='bold')

ax.set_ylim(0.80, 1.05)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Métrica')
ax.set_title('Comparación: Árbol Completo vs Árbol Optimizado')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/08_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Guardado: outputs/08_comparison.png")

# %% [markdown]
# ## 9. Resumen de Resultados

# %%
# ==============================
# 8. RESUMEN FINAL
# ==============================
print("\n" + "=" * 65)
print("RESUMEN DE RESULTADOS")
print("=" * 65)
print(f"\n{'Modelo':<25} {'Accuracy':>10} {'F1 Score':>10} {'Profundidad':>12} {'Hojas':>6}")
print("-" * 65)
print(f"{'Árbol sin podado':<25} {acc_full:>10.4f} {f1_full:>10.4f} "
      f"{dt_full.get_depth():>12} {dt_full.get_n_leaves():>6}")
print(f"{'Árbol optimizado':<25} {acc_best:>10.4f} {f1_best:>10.4f} "
      f"{dt_best.get_depth():>12} {dt_best.get_n_leaves():>6}")

top_feature = feature_names[np.argmax(dt_best.feature_importances_)]
top_importance = dt_best.feature_importances_.max()
print(f"\nCaracterística más importante: {top_feature}")
print(f"  Importancia: {top_importance:.4f}")
print(f"\nMejores hiperparámetros: {grid_search.best_params_}")
print(f"Cross-Validation F1 (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

print("\nArchivos generados en outputs/:")
for f in sorted(os.listdir('outputs')):
    print(f"  {f}")
