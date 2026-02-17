# =============================================================
# LABORATORIO DE PREPROCESAMIENTO Y EXPLORACIÓN INICIAL (FINANZAS)
# =============================================================
#
# En el ciclo de vida de ML, antes de entrenar cualquier modelo,
# es fundamental:
# - Entender los datos
# - Limpiarlos
# - Prepararlos correctamente
# - Separarlos en entrenamiento y prueba
#
# Este archivo se enfoca JUSTAMENTE en esas etapas previas al modelado.
#
# Flujo general (pensado como ML Lifecycle):
# 1. Ingesta de datos (cargar el CSV)
# 2. Comprensión de los datos (EDA)
# 3. Preparación de datos (limpieza y transformación)
# 4. Feature engineering (crear nuevas variables)
# 5. Split train / test (simular datos futuros)
# 6. Escalado sin fuga de información
# 7. Guardado de datasets listos para entrenar modelos

import argparse
from pathlib import Path
import json
import warnings

# Para no distraernos con advertencias durante el laboratorio.
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# Utilidades típicas del ciclo de vida de ML:
# - Separar datos en entrenamiento y prueba
# - Escalar variables numéricas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------
# FUNCIONES AUXILIARES
# -------------------------------------------------------------
# Estas funciones representan pequeñas tareas repetitivas del
# proceso de preparación de datos.


def try_read_csv(path, sep, encoding):
    """
    Fase del ciclo ML: INGESTA DE DATOS

    Intenta cargar el archivo CSV de forma robusta.
    Si el archivo tiene problemas de codificación,
    se prueba automáticamente con otra alternativa.

    La idea es que el pipeline no se rompa solo por
    el formato del archivo.
    """
    try:
        return pd.read_csv(path, sep=sep, encoding=encoding)
    except UnicodeDecodeError:
        print("[WARN] Problema de encoding. Reintentando con 'latin-1'.")
        return pd.read_csv(path, sep=sep, encoding="latin-1")


def winsorize_df(df, numeric_cols, lower_q=0.01, upper_q=0.99):
    """
    Fase del ciclo ML: LIMPIEZA DE DATOS

    En lugar de eliminar valores extremos (outliers),
    los limitamos a un rango razonable.

    Esto ayuda a que el modelo no se vea dominado
    por valores atípicos muy grandes o muy pequeños.
    """
    for c in numeric_cols:
        lower = df[c].quantile(lower_q)
        upper = df[c].quantile(upper_q)
        df[c] = df[c].clip(lower, upper)
    return df


def iqr_outlier_mask(series):
    """
    Fase del ciclo ML: ANÁLISIS DE DATOS

    Identifica posibles outliers usando una regla sencilla:
    comparar los valores con el rango típico de la variable.

    Aquí solo los detectamos para reportarlos,
    no para eliminarlos automáticamente.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (series < lower) | (series > upper)


def print_section(title):
    """
    Función de apoyo para mostrar claramente
    en qué etapa del pipeline estamos.
    """
    print("=" * 70)
    print(title)
    print("=" * 70)


def parse_list_arg(arg):
    """
    Convierte parámetros escritos en la terminal
    (por ejemplo: "A,B,C") en listas de Python.

    Esto permite que el usuario configure el pipeline
    sin tocar el código.
    """
    if arg is None or len(arg.strip()) == 0:
        return []
    return [a.strip() for a in arg.split(",") if a.strip()]


# -------------------------------------------------------------
# CONFIGURACIÓN DEL PIPELINE (CLI)
# -------------------------------------------------------------
# Esta sección corresponde a la fase de DISEÑO DEL PIPELINE.
# Aquí se define qué puede cambiar el usuario al ejecutar el script.

parser = argparse.ArgumentParser(
    description="Pipeline de preparación de datos financieros (ML Lifecycle)"
)

parser.add_argument("--input", required=True, help="Ruta al archivo CSV")
parser.add_argument("--sep", default=",", help="Separador del CSV")
parser.add_argument("--encoding", default="utf-8", help="Encoding del archivo")

parser.add_argument("--date-col", default=None, help="Columna de fecha")
parser.add_argument("--id-cols", default="", help="Columnas identificadoras")
parser.add_argument("--categorical-cols", default="", help="Columnas categóricas")
parser.add_argument("--numeric-cols", default="", help="Columnas numéricas")
parser.add_argument("--price-cols", default="", help="Columnas de precios")
parser.add_argument("--target-col", default=None, help="Variable objetivo")

parser.add_argument("--missing-tokens", default="NA,N/A,na,NaN,?,-999,",
                    help="Valores que se interpretan como faltantes")

parser.add_argument("--time-split", action="store_true",
                    help="Usar división temporal (pasado vs futuro)")
parser.add_argument("--split-date", default=None, help="Fecha de corte")
parser.add_argument("--test-size", type=float, default=0.2,
                    help="Porcentaje del conjunto de prueba")

parser.add_argument("--winsorize", nargs=2, type=float, default=None,
                    help="Limitar outliers usando cuantiles")

parser.add_argument("--outdir", default="./data_output_finanzas",
                    help="Directorio de salida")

args = parser.parse_args()

# =============================================================
# INICIO DEL CICLO DE VIDA DE ML
# =============================================================

# -------------------------------------------------------------
# 1) INGESTA DE DATOS
# -------------------------------------------------------------
print_section("1) INGESTA DE DATOS")

input_path = Path(args.input)
if not input_path.exists():
    raise FileNotFoundError(f"No se encontró el archivo: {input_path}")

missing_values = parse_list_arg(args.missing_tokens)

# Cargamos el dataset crudo
df = try_read_csv(input_path, sep=args.sep, encoding=args.encoding)

# Limpieza básica de texto:
# Esto evita problemas silenciosos más adelante
for c in df.select_dtypes(include=["object", "string"]).columns:
    df[c] = df[c].astype(str).str.strip()
    df[c] = df[c].replace({tok: np.nan for tok in missing_values})

print("Dimensiones del dataset:", df.shape)

# -------------------------------------------------------------
# 2) ENTENDIMIENTO DE LOS DATOS (EDA)
# -------------------------------------------------------------
print_section("2) ENTENDIMIENTO DE LOS DATOS")

# Convertimos fechas si existen
if args.date_col and args.date_col in df.columns:
    df[args.date_col] = pd.to_datetime(df[args.date_col], errors="coerce")

id_cols = parse_list_arg(args.id_cols)
cat_cols = parse_list_arg(args.categorical_cols)
num_cols = parse_list_arg(args.numeric_cols)
price_cols = parse_list_arg(args.price_cols)

# Si el usuario no define columnas numéricas, las inferimos automáticamente
if not num_cols:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Si no define categóricas, tomamos las de tipo texto
if not cat_cols:
    exclude = set(id_cols + ([args.date_col] if args.date_col else []) +
                  ([args.target_col] if args.target_col else []))
    cat_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns if c not in exclude]

print("Variables numéricas:", num_cols)
print("Variables categóricas:", cat_cols)

print(df[num_cols].describe().T)
print("Valores nulos por columna:")
print(df.isna().sum().sort_values(ascending=False).head(10))

# -------------------------------------------------------------
# 3) PREPARACIÓN DE DATOS
# -------------------------------------------------------------
print_section("3) PREPARACIÓN DE DATOS")

# Imputación sencilla:
# En esta etapa buscamos un baseline, no la imputación perfecta
for c in num_cols:
    df[c] = df[c].fillna(df[c].median())

for c in cat_cols:
    df[c] = df[c].fillna("__MISSING__")

# Detección de outliers (solo informativo)
outliers = {c: int(iqr_outlier_mask(df[c]).sum()) for c in num_cols}
print("Outliers detectados (referencia):", outliers)

# Tratamiento opcional de outliers
if args.winsorize:
    df[num_cols] = winsorize_df(df[num_cols], num_cols,
                                args.winsorize[0], args.winsorize[1])

# -------------------------------------------------------------
# 4) FEATURE ENGINEERING
# -------------------------------------------------------------
print_section("4) FEATURE ENGINEERING")

# Crear variables que el modelo pueda aprender mejor
new_feats = []

if price_cols:
    if args.date_col:
        df = df.sort_values(([args.date_col] if not id_cols else id_cols + [args.date_col]))

    for pc in price_cols:
        df[pc + "_ret"] = df.groupby(id_cols)[pc].pct_change() if id_cols else df[pc].pct_change()
        df[pc + "_logret"] = np.log1p(df[pc + "_ret"])
        df[[pc + "_ret", pc + "_logret"]] = df[[pc + "_ret", pc + "_logret"]].fillna(0.0)
        new_feats.extend([pc + "_ret", pc + "_logret"])

num_cols = list(set(num_cols + new_feats))
print("Nuevas variables creadas:", new_feats)

# -------------------------------------------------------------
# 5) SPLIT TRAIN / TEST (SIMULAR EL FUTURO)
# -------------------------------------------------------------
print_section("5) TRAIN / TEST SPLIT")

if args.target_col and args.target_col in df.columns:
    y = df[args.target_col]
    X = df.drop(columns=[args.target_col])
else:
    y = None
    X = df.copy()

# Eliminamos columnas que no deben aprenderse directamente
for c in id_cols + ([args.date_col] if args.date_col else []):
    if c in X.columns:
        X = X.drop(columns=[c])

# Convertimos categorías a números
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

num_in_X = [c for c in num_cols if c in X.columns]
scaler = StandardScaler()

# División temporal o aleatoria
if args.time_split and args.date_col:
    cutoff = pd.to_datetime(args.split_date)
    X_train, X_test = X[df[args.date_col] < cutoff], X[df[args.date_col] >= cutoff]
    y_train = y[df[args.date_col] < cutoff] if y is not None else None
    y_test = y[df[args.date_col] >= cutoff] if y is not None else None
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    ) if y is not None else (*train_test_split(X, test_size=args.test_size, random_state=42), None, None)

# -------------------------------------------------------------
# 6) ESCALADO (SIN HACER TRAMPA)
# -------------------------------------------------------------
print_section("6) ESCALADO")

# El modelo solo puede aprender estadísticas del conjunto de entrenamiento
X_train[num_in_X] = scaler.fit_transform(X_train[num_in_X])
X_test[num_in_X] = scaler.transform(X_test[num_in_X])

# -------------------------------------------------------------
# 7) EXPORTACIÓN
# -------------------------------------------------------------
print_section("7) EXPORTACIÓN")

outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)

train_df = X_train.assign(**({args.target_col: y_train} if y_train is not None else {}))
test_df = X_test.assign(**({args.target_col: y_test} if y_test is not None else {}))

train_df.to_parquet(outdir / "finance_train.parquet", index=False)
test_df.to_parquet(outdir / "finance_test.parquet", index=False)

# Guardamos metadatos para reproducibilidad
metadata = {
    "source": str(input_path),
    "features_numeric": num_cols,
    "features_categorical": cat_cols,
    "features_generated": new_feats,
    "train_shape": train_df.shape,
    "test_shape": test_df.shape
}

with open(outdir / "finance_data_dictionary.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("Pipeline de ML finalizado correctamente.")