# ============================================================================
# 1. IMPORTACIONES
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import reciprocal, uniform
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn: Modelos
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

# Scikit-learn: Métricas
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, f1_score, confusion_matrix, 
    classification_report, roc_auc_score, roc_curve
)

# Configuración visual
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 10

# Configuración de reproducibilidad
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ============================================================================
# 2. FUNCIONES AUXILIARES
# ============================================================================

def print_section(title):
    """Imprime un encabezado de sección con formato."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def print_subsection(title):
    """Imprime un subencabezado con formato."""
    print(f"\n{'─'*80}")
    print(f"  {title}")
    print(f"{'─'*80}")


def load_and_clean_data(filepath):
    """
    Carga y limpia el dataset de películas.
    
    Parameters:
    -----------
    filepath : str
        Ruta al archivo CSV
        
    Returns:
    --------
    df : pd.DataFrame
        Dataset limpio y procesado
    """
    print_section("1. CARGA Y LIMPIEZA DE DATOS")
    
    # Cargar datos
    df = pd.read_csv(filepath)
    print(f"✓ Dataset cargado: {df.shape[0]} películas, {df.shape[1]} columnas")
    
    # Mostrar información inicial
    print(f"\nColumnas disponibles:")
    print(f"  {', '.join(df.columns.tolist())}")
    
    print(f"\nEstadísticas iniciales:")
    print(f"  Valores faltantes: {df.isnull().sum().sum()}")
    print(f"  Duplicados: {df.duplicated().sum()}")
    
    # Limpieza específica para este dataset
    # Convertir columnas numéricas
    df['RATING'] = pd.to_numeric(df['RATING'], errors='coerce')
    df['RunTime'] = pd.to_numeric(df['RunTime'], errors='coerce')
    
    # Convertir VOTES y Gross (remover símbolos, etc.)
    df['VOTES'] = df['VOTES'].astype(str).str.replace(',', '').str.replace('K', '000')
    df['VOTES'] = pd.to_numeric(df['VOTES'], errors='coerce')
    
    df['Gross'] = df['Gross'].astype(str).str.replace('$', '').str.replace('M', '')
    df['Gross'] = pd.to_numeric(df['Gross'], errors='coerce')
    
    # Extraer año (remover paréntesis)
    df['YEAR'] = df['YEAR'].astype(str).str.replace('(', '').str.replace(')', '')
    df['YEAR'] = df['YEAR'].str.split('–').str[0].str.strip()
    df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce')
    
    # Eliminar filas con valores faltantes críticos
    df_clean = df.dropna(subset=['RATING', 'RunTime', 'VOTES', 'Gross', 'YEAR'])
    
    print(f"\n✓ Datos limpios: {df_clean.shape[0]} películas")
    print(f"  Filas eliminadas: {df.shape[0] - df_clean.shape[0]}")
    
    # Crear variable binaria para rentabilidad
    # Consideramos una película rentable si tiene buena rating y muchos votos
    df_clean['Profitable'] = (
        (df_clean['RATING'] >= df_clean['RATING'].median()) & 
        (df_clean['VOTES'] >= df_clean['VOTES'].median())
    ).astype(int)
    
    print(f"\nVariable 'Profitable' (para regresión logística):")
    print(f"  Películas rentables: {df_clean['Profitable'].sum()}")
    print(f"  Películas no rentables: {(1 - df_clean['Profitable']).sum()}")
    
    return df_clean


def exploratory_data_analysis(df):
    """
    Realiza análisis exploratorio de datos.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset a analizar
    """
    print_section("2. ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
    
    # Estadísticas descriptivas
    print("\nEstadísticas Descriptivas:")
    print(df[['RATING', 'RunTime', 'VOTES', 'Gross', 'YEAR']].describe())
    
    # Matriz de correlación
    print_subsection("Matriz de Correlación")
    numeric_cols = ['RATING', 'RunTime', 'VOTES', 'Gross', 'YEAR', 'Profitable']
    correlation_matrix = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, ax=ax, cbar_kws={'label': 'Correlación'})
    ax.set_title('Matriz de Correlación - Dataset de Películas', 
                 fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('outputs/01_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfico guardado: 01_correlation_matrix.png")
    
    # Distribuciones
    print_subsection("Distribuciones de Variables")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(['RATING', 'RunTime', 'VOTES', 'Gross', 'YEAR']):
        axes[idx].hist(df[col], bins=30, color='#2E86AB', edgecolor='black', alpha=0.7)
        axes[idx].set_xlabel(col, fontweight='bold')
        axes[idx].set_ylabel('Frecuencia', fontweight='bold')
        axes[idx].set_title(f'Distribución de {col}', fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
    
    # Distribución de variable objetivo
    categories = ['No Rentable', 'Rentable']
    colors = ['#FF6B6B', '#4ECDC4']
    values = df['Profitable'].value_counts().values
    axes[5].bar(categories, values, color=colors, edgecolor='black', width=0.6)
    axes[5].set_ylabel('Cantidad', fontweight='bold')
    axes[5].set_title('Distribución de Profitable', fontweight='bold')
    axes[5].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(values):
        axes[5].text(i, v + max(values)*0.02, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/02_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfico guardado: 02_distributions.png")


# ============================================================================
# 3. REGRESIÓN LINEAL
# ============================================================================

def linear_regression_analysis(df):
    """
    Realiza análisis completo de regresión lineal.
    Predice: Gross (Ingresos) usando variables de presupuesto/popularidad
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset limpio
    """
    print_section("3. REGRESIÓN LINEAL - PREDICCIÓN DE INGRESOS (GROSS)")
    
    # ========================================================================
    # 3.1 Preparación de datos
    # ========================================================================
    print_subsection("3.1 Preparación de Datos")
    
    # Seleccionar features y target
    feature_columns = ['RATING', 'RunTime', 'VOTES', 'YEAR']
    X = df[feature_columns].copy()
    y = df['Gross'].copy()
    
    print(f"Features: {feature_columns}")
    print(f"Target: Gross (Ingresos en millones)")
    print(f"Dimensiones X: {X.shape}")
    print(f"Dimensiones y: {y.shape}")
    
    # División entrenamiento/prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE
    )
    
    print(f"\n✓ División de datos:")
    print(f"  Entrenamiento: {X_train.shape[0]} muestras (70%)")
    print(f"  Prueba: {X_test.shape[0]} muestras (30%)")
    
    # Visualizar conjuntos
    print_subsection("Visualización de Conjuntos")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Scatter plot: RATING vs Gross
    axes[0].scatter(X_train['RATING'], y_train, alpha=0.6, s=50, 
                   label='Entrenamiento', color='#2E86AB', edgecolors='black', linewidth=0.5)
    axes[0].scatter(X_test['RATING'], y_test, alpha=0.6, s=50, 
                   label='Prueba', color='#A23B72', edgecolors='black', linewidth=0.5)
    axes[0].set_xlabel('RATING', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Gross (millones USD)', fontsize=11, fontweight='bold')
    axes[0].set_title('Gross vs Rating - Conjuntos de Entrenamiento y Prueba', 
                     fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Distribuciones de y
    axes[1].hist(y_train, alpha=0.6, bins=20, label='Entrenamiento', color='#2E86AB', edgecolor='black')
    axes[1].hist(y_test, alpha=0.6, bins=20, label='Prueba', color='#A23B72', edgecolor='black')
    axes[1].set_xlabel('Gross (millones USD)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
    axes[1].set_title('Distribución de Gross', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('outputs/03_linear_regression_train_test.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfico guardado: 03_linear_regression_train_test.png")
    
    # ========================================================================
    # 3.2 Definición de pipelines
    # ========================================================================
    print_subsection("3.2 Definición de Pipelines")
    
    # Pipeline para Ridge
    pipeline_ridge = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge())
    ])
    print("✓ Pipeline Ridge: StandardScaler → Ridge")
    
    # Pipeline para Lasso
    pipeline_lasso = Pipeline([
        ('scaler', StandardScaler()),
        ('lasso', Lasso(max_iter=10000))
    ])
    print("✓ Pipeline Lasso: StandardScaler → Lasso")
    
    # ========================================================================
    # 3.3 Búsqueda de hiperparámetros
    # ========================================================================
    print_subsection("3.3 Búsqueda Aleatoria de Hiperparámetros + Cross-Validation")
    
    # Distribuciones de parámetros
    param_dist_ridge = {
        'ridge__alpha': reciprocal(0.001, 100)
    }
    
    param_dist_lasso = {
        'lasso__alpha': reciprocal(0.0001, 10)
    }
    
    print("Buscando mejores parámetros para Ridge...")
    random_search_ridge = RandomizedSearchCV(
        pipeline_ridge,
        param_dist_ridge,
        n_iter=20,
        cv=5,
        scoring='r2',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )
    random_search_ridge.fit(X_train, y_train)
    print(f"✓ Ridge - Mejores parámetros: {random_search_ridge.best_params_}")
    print(f"  R² CV: {random_search_ridge.best_score_:.4f}")
    
    print("\nBuscando mejores parámetros para Lasso...")
    random_search_lasso = RandomizedSearchCV(
        pipeline_lasso,
        param_dist_lasso,
        n_iter=20,
        cv=5,
        scoring='r2',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )
    random_search_lasso.fit(X_train, y_train)
    print(f"✓ Lasso - Mejores parámetros: {random_search_lasso.best_params_}")
    print(f"  R² CV: {random_search_lasso.best_score_:.4f}")
    
    best_ridge = random_search_ridge.best_estimator_
    best_lasso = random_search_lasso.best_estimator_
    
    # ========================================================================
    # 3.4 Evaluación
    # ========================================================================
    print_subsection("3.4 Evaluación en Conjunto de Prueba")
    
    # Predicciones
    y_pred_ridge = best_ridge.predict(X_test)
    y_pred_lasso = best_lasso.predict(X_test)
    
    # Métricas Ridge
    r2_ridge = r2_score(y_test, y_pred_ridge)
    mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
    rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
    
    # Métricas Lasso
    r2_lasso = r2_score(y_test, y_pred_lasso)
    mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
    rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
    
    print("\nRIDGE (L2 Regularization):")
    print(f"  R² Score:  {r2_ridge:.4f}")
    print(f"  MAE:       ${mae_ridge:.2f}M")
    print(f"  RMSE:      ${rmse_ridge:.2f}M")
    
    print("\nLASSO (L1 Regularization):")
    print(f"  R² Score:  {r2_lasso:.4f}")
    print(f"  MAE:       ${mae_lasso:.2f}M")
    print(f"  RMSE:      ${rmse_lasso:.2f}M")
    
    # Mejor modelo
    mejor_modelo = 'Ridge' if r2_ridge > r2_lasso else 'Lasso'
    print(f"\n✓ Mejor modelo: {mejor_modelo}")
    print(f"  Diferencia en R²: {abs(r2_ridge - r2_lasso):.4f}")
    
    # ========================================================================
    # 3.5 Visualización de resultados
    # ========================================================================
    print_subsection("3.5 Visualización de Resultados")
    
    # Gráfico: Predicción vs Real
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Ridge
    axes[0].scatter(y_test, y_pred_ridge, alpha=0.6, s=60, color='#2E86AB', 
                   edgecolors='black', linewidth=0.5)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                r'--', lw=2, color='red', label='Predicción Perfecta')
    axes[0].set_xlabel('Gross Real (millones USD)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Gross Predicho (millones USD)', fontsize=11, fontweight='bold')
    axes[0].set_title(f'Ridge - R² = {r2_ridge:.4f}', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Lasso
    axes[1].scatter(y_test, y_pred_lasso, alpha=0.6, s=60, color='#A23B72', 
                   edgecolors='black', linewidth=0.5)
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                r'--', lw=2, color='red', label='Predicción Perfecta')
    axes[1].set_xlabel('Gross Real (millones USD)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Gross Predicho (millones USD)', fontsize=11, fontweight='bold')
    axes[1].set_title(f'Lasso - R² = {r2_lasso:.4f}', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/04_linear_regression_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfico guardado: 04_linear_regression_predictions.png")
    
    # Gráfico: Residuos
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    residuos_ridge = y_test - y_pred_ridge
    residuos_lasso = y_test - y_pred_lasso
    
    # Ridge
    axes[0].scatter(y_pred_ridge, residuos_ridge, alpha=0.6, s=60, color='#2E86AB', 
                   edgecolors='black', linewidth=0.5)
    axes[0].axhline(y=0, color='red', linestyle='--', lw=2)
    axes[0].set_xlabel('Gross Predicho (millones USD)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Residuos', fontsize=11, fontweight='bold')
    axes[0].set_title('Ridge - Gráfico de Residuos', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Lasso
    axes[1].scatter(y_pred_lasso, residuos_lasso, alpha=0.6, s=60, color='#A23B72', 
                   edgecolors='black', linewidth=0.5)
    axes[1].axhline(y=0, color='red', linestyle='--', lw=2)
    axes[1].set_xlabel('Gross Predicho (millones USD)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Residuos', fontsize=11, fontweight='bold')
    axes[1].set_title('Lasso - Gráfico de Residuos', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/05_linear_regression_residuals.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfico guardado: 05_linear_regression_residuals.png")
    
    # Gráfico: Comparación de modelos
    fig, ax = plt.subplots(figsize=(10, 6))
    
    modelos = ['Ridge', 'Lasso']
    r2_scores = [r2_ridge, r2_lasso]
    mae_scores = [mae_ridge, mae_lasso]
    
    x = np.arange(len(modelos))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, r2_scores, width, label='R² Score', 
                  color='#2E86AB', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, [m/max(mae_scores) for m in mae_scores], width, 
                  label='MAE (normalizado)', color='#A23B72', edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Valor', fontsize=12, fontweight='bold')
    ax.set_title('Comparación de Modelos - Regresión Lineal', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(modelos, fontsize=11, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(r2_scores) * 1.15)
    
    # Agregar valores en barras
    for bar, val in zip(bars1, r2_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    for bar, val in zip(bars2, mae_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'${val:.1f}M', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('outputs/06_linear_regression_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfico guardado: 06_linear_regression_comparison.png")
    
    return {
        'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test,
        'best_ridge': best_ridge, 'best_lasso': best_lasso,
        'r2_ridge': r2_ridge, 'r2_lasso': r2_lasso,
        'mae_ridge': mae_ridge, 'mae_lasso': mae_lasso
    }


# ============================================================================
# 4. REGRESIÓN LOGÍSTICA
# ============================================================================

def logistic_regression_analysis(df):
    """
    Realiza análisis completo de regresión logística.
    Predice: Profitable (Rentabilidad) - Clasificación binaria
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset limpio
    """
    print_section("4. REGRESIÓN LOGÍSTICA - PREDICCIÓN DE RENTABILIDAD")
    
    # ========================================================================
    # 4.1 Preparación de datos
    # ========================================================================
    print_subsection("4.1 Preparación de Datos")
    
    # Seleccionar features y target
    feature_columns = ['RATING', 'RunTime', 'VOTES', 'YEAR', 'Gross']
    X = df[feature_columns].copy()
    y = df['Profitable'].copy()
    
    print(f"Features: {feature_columns}")
    print(f"Target: Profitable (0=No rentable, 1=Rentable)")
    print(f"Dimensiones X: {X.shape}")
    print(f"Dimensiones y: {y.shape}")
    
    print(f"\nDistribución de clases:")
    print(f"  No rentable (0): {(y == 0).sum()} películas ({(y == 0).sum()/len(y)*100:.1f}%)")
    print(f"  Rentable (1): {(y == 1).sum()} películas ({(y == 1).sum()/len(y)*100:.1f}%)")
    
    # División entrenamiento/prueba (mantener proporciones)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"\n✓ División de datos:")
    print(f"  Entrenamiento: {X_train.shape[0]} muestras (70%)")
    print(f"  Prueba: {X_test.shape[0]} muestras (30%)")
    
    # Visualizar conjuntos
    print_subsection("Visualización de Conjuntos")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Distribución de clases
    y_train_counts = y_train.value_counts()
    y_test_counts = y_test.value_counts()
    
    x = np.arange(2)
    width = 0.35
    
    axes[0].bar(x - width/2, [y_train_counts[0], y_train_counts[1]], width, 
               label='Entrenamiento', color='#2E86AB', edgecolor='black', linewidth=1.5)
    axes[0].bar(x + width/2, [y_test_counts[0], y_test_counts[1]], width, 
               label='Prueba', color='#A23B72', edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Cantidad de Películas', fontsize=11, fontweight='bold')
    axes[0].set_title('Distribución de Clases', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['No Rentable', 'Rentable'], fontsize=10, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Proporciones en pie chart
    labels = ['No Rentable', 'Rentable']
    colors = ['#FF6B6B', '#4ECDC4']
    sizes = y.value_counts().values
    
    axes[1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
               startangle=90, explode=(0.05, 0.05), 
               textprops={'fontweight': 'bold', 'fontsize': 11})
    axes[1].set_title('Proporción de Clases (Dataset Completo)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/07_logistic_regression_classes.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfico guardado: 07_logistic_regression_classes.png")
    
    # ========================================================================
    # 4.2 Pipeline y búsqueda de hiperparámetros
    # ========================================================================
    print_subsection("4.2 Pipeline y Búsqueda de Hiperparámetros")
    
    # Pipeline
    pipeline_logistic = Pipeline([
        ('scaler', StandardScaler()),
        ('logistic', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ])
    print("✓ Pipeline Logístico: StandardScaler → LogisticRegression")
    
    # Distribuciones de parámetros
    param_dist_logistic = {
        'logistic__C': reciprocal(0.01, 100),
        'logistic__penalty': ['l2']
    }
    
    print("\nBuscando mejores parámetros para Regresión Logística...")
    random_search_logistic = RandomizedSearchCV(
        pipeline_logistic,
        param_dist_logistic,
        n_iter=20,
        cv=5,
        scoring='f1',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )
    random_search_logistic.fit(X_train, y_train)
    print(f"✓ Mejores parámetros: {random_search_logistic.best_params_}")
    print(f"  F1 Score CV: {random_search_logistic.best_score_:.4f}")
    
    best_logistic = random_search_logistic.best_estimator_
    
    # ========================================================================
    # 4.3 Evaluación
    # ========================================================================
    print_subsection("4.3 Evaluación en Conjunto de Prueba")
    
    # Predicciones
    y_pred_logistic = best_logistic.predict(X_test)
    y_pred_proba_logistic = best_logistic.predict_proba(X_test)[:, 1]
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred_logistic)
    f1 = f1_score(y_test, y_pred_logistic)
    roc_auc = roc_auc_score(y_test, y_pred_proba_logistic)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")
    
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred_logistic, 
                               target_names=['No Rentable', 'Rentable']))
    
    # ========================================================================
    # 4.4 Visualización de resultados
    # ========================================================================
    print_subsection("4.4 Visualización de Resultados")
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred_logistic)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax,
               xticklabels=['No Rentable', 'Rentable'],
               yticklabels=['No Rentable', 'Rentable'],
               annot_kws={'fontsize': 14, 'fontweight': 'bold'})
    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicho', fontsize=12, fontweight='bold')
    ax.set_title(f'Matriz de Confusión - Regresión Logística\nAccuracy: {accuracy:.4f}', 
                fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('outputs/08_logistic_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfico guardado: 08_logistic_confusion_matrix.png")
    
    # Curva ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_logistic)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(fpr, tpr, color='#2E86AB', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Clasificador Aleatorio')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('Curva ROC - Regresión Logística', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/09_logistic_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfico guardado: 09_logistic_roc_curve.png")
    
    # Distribución de probabilidades predichas
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(y_pred_proba_logistic[y_test == 0], bins=20, alpha=0.6, 
           label='No Rentables', color='#FF6B6B', edgecolor='black')
    ax.hist(y_pred_proba_logistic[y_test == 1], bins=20, alpha=0.6, 
           label='Rentables', color='#4ECDC4', edgecolor='black')
    ax.axvline(x=0.5, color='red', linestyle='--', lw=2.5, label='Threshold = 0.5')
    ax.set_xlabel('Probabilidad Predicha', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
    ax.set_title('Distribución de Probabilidades Predichas', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('outputs/10_logistic_probability_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfico guardado: 10_logistic_probability_distribution.png")
    
    return {
        'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test,
        'best_logistic': best_logistic,
        'accuracy': accuracy, 'f1': f1, 'roc_auc': roc_auc,
        'cm': cm, 'fpr': fpr, 'tpr': tpr
    }


# ============================================================================
# 5. RESUMEN FINAL
# ============================================================================

def print_summary(lr_results, logistic_results):
    """
    Imprime un resumen final de todos los resultados.
    
    Parameters:
    -----------
    lr_results : dict
        Resultados de regresión lineal
    logistic_results : dict
        Resultados de regresión logística
    """
    print_section("5. RESUMEN FINAL")
    
    print("\nREGRESIÓN LINEAL (Predicción de Ingresos - Gross)")
    print("─" * 80)
    print(f"Ridge R² Score:    {lr_results['r2_ridge']:.4f}")
    print(f"Ridge MAE:         ${lr_results['mae_ridge']:.2f}M")
    print(f"\nLasso R² Score:    {lr_results['r2_lasso']:.4f}")
    print(f"Lasso MAE:         ${lr_results['mae_lasso']:.2f}M")
    
    mejor_lr = 'Ridge' if lr_results['r2_ridge'] > lr_results['r2_lasso'] else 'Lasso'
    print(f"\n✓ Mejor modelo: {mejor_lr}")
    
    print("\n\nREGRESIÓN LOGÍSTICA (Predicción de Rentabilidad)")
    print("─" * 80)
    print(f"Accuracy:          {logistic_results['accuracy']:.4f}")
    print(f"F1 Score:          {logistic_results['f1']:.4f}")
    print(f"ROC AUC:           {logistic_results['roc_auc']:.4f}")
    
    print("\nMatriz de Confusión:")
    cm = logistic_results['cm']
    print(f"  Verdaderos Negativos:  {cm[0, 0]}")
    print(f"  Falsos Positivos:      {cm[0, 1]}")
    print(f"  Falsos Negativos:      {cm[1, 0]}")
    print(f"  Verdaderos Positivos:  {cm[1, 1]}")
    
    print("\n\nARCHIVOS GENERADOS")
    print("─" * 80)
    print("✓ 01_correlation_matrix.png")
    print("✓ 02_distributions.png")
    print("✓ 03_linear_regression_train_test.png")
    print("✓ 04_linear_regression_predictions.png")
    print("✓ 05_linear_regression_residuals.png")
    print("✓ 06_linear_regression_comparison.png")
    print("✓ 07_logistic_regression_classes.png")
    print("✓ 08_logistic_confusion_matrix.png")
    print("✓ 09_logistic_roc_curve.png")
    print("✓ 10_logistic_probability_distribution.png")
    
    print("\n" + "="*80)
    print("  ✓ ANÁLISIS COMPLETADO EXITOSAMENTE")
    print("="*80 + "\n")


# ============================================================================
# 6. FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal que coordina todo el análisis."""
    
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "  ANÁLISIS DE REGRESIÓN LINEAL Y LOGÍSTICA - DATASET DE PELÍCULAS".center(78) + "║")
    print("║" + "  SI3015 - Fundamentos de Aprendizaje Automático".center(78) + "║")
    print("║" + "  Semana 5 - Ejercicio Práctico".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    
    # Cargar y limpiar datos
    df = load_and_clean_data('movies.csv')
    
    # EDA
    exploratory_data_analysis(df)
    
    # Regresión lineal
    lr_results = linear_regression_analysis(df)
    
    # Regresión logística
    logistic_results = logistic_regression_analysis(df)
    
    # Resumen final
    print_summary(lr_results, logistic_results)


if __name__ == "__main__":
    main()