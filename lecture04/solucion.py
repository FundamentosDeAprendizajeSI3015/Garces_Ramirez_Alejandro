import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler


# ============================================================
# 1) CARGAR EL CONJUNTO DE DATOS Y EXPLORACIÓN INICIAL
# ============================================================

df = pd.read_csv("movies.csv")

print("\n============================")
print("1) EXPLORACIÓN INICIAL")
print("============================")
print("\n--- HEAD ---")
print(df.head())
print("\n--- INFO (antes de limpiar) ---")
print(df.info())


# ============================================================
# 2) LIMPIEZA (la mayor posible)
#    - YEAR: extrae el primer año aunque venga como rango (2010–2022)
#    - VOTES: quita comas, convierte a numérico
#    - Gross: soporta $, €, £, comas, y sufijos K/M/B
# ============================================================

print("\n============================")
print("2) LIMPIEZA DE DATOS")
print("============================")

# YEAR: extraer el primer año que aparezca (maneja (2021– ), (2010–2022), (2021), etc.)
df["YEAR"] = (
    df["YEAR"]
    .astype(str)
    .str.extract(r"(\d{4})")[0]
)
df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")

# RATING: numérico
df["RATING"] = pd.to_numeric(df["RATING"], errors="coerce")

# VOTES: quitar comas y convertir a numérico
df["VOTES"] = (
    df["VOTES"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.strip()
)
df["VOTES"] = pd.to_numeric(df["VOTES"], errors="coerce")

# RunTime: numérico
df["RunTime"] = pd.to_numeric(df["RunTime"], errors="coerce")

# GENRE: limpiar saltos de línea / espacios extremos
df["GENRE"] = df["GENRE"].astype(str).str.replace("\n", " ").str.strip()

# MOVIES: limpiar espacios
df["MOVIES"] = df["MOVIES"].astype(str).str.strip()

# Gross: parser robusto
def parse_gross(x):
    if pd.isna(x):
        return np.nan

    s = str(x).strip()

    # Vacíos comunes
    if s == "" or s.lower() in {"nan", "none", "-"}:
        return np.nan

    # Quitar símbolos y comas
    s = s.replace("$", "").replace("€", "").replace("£", "").replace(",", "").strip()

    # Sufijos K/M/B
    mult = 1
    if s.endswith(("K", "k")):
        mult = 1_000
        s = s[:-1]
    elif s.endswith(("M", "m")):
        mult = 1_000_000
        s = s[:-1]
    elif s.endswith(("B", "b")):
        mult = 1_000_000_000
        s = s[:-1]

    try:
        return float(s) * mult
    except ValueError:
        return np.nan

df["Gross"] = df["Gross"].apply(parse_gross)

# Eliminar filas totalmente vacías (por seguridad)
df.dropna(how="all", inplace=True)

print("\n--- INFO (después de limpiar columnas clave) ---")
print(df.info())

# Conteo de no nulos por columna
print("\n--- NULOS / NO NULOS ---")
print(df.isna().sum().sort_values(ascending=False))
print("\n--- No nulos Gross (para verificar que no quedó vacío) ---")
print("Gross no nulos:", df["Gross"].notna().sum())


# ============================================================
# 3) MEDIDAS: TENDENCIA CENTRAL, DISPERSIÓN, POSICIÓN
# ============================================================

print("\n============================")
print("3) MEDIDAS ESTADÍSTICAS")
print("============================")

cols_num = ["RATING", "VOTES", "RunTime", "Gross", "YEAR"]

# ---------- Tendencia central (sin error de mode) ----------
mean_vals = df[cols_num].mean(numeric_only=True)
median_vals = df[cols_num].median(numeric_only=True)

mode_df = df[cols_num].mode(numeric_only=True)
if mode_df.empty:
    mode_vals = pd.Series([np.nan] * len(cols_num), index=cols_num)
else:
    mode_vals = mode_df.iloc[0]

central_tendency = pd.DataFrame({
    "mean": mean_vals,
    "median": median_vals,
    "mode": mode_vals
})

print("\n--- Medidas de Tendencia Central ---")
print(central_tendency)

# ---------- Dispersión ----------
dispersion = df[cols_num].agg(["std", "var", "min", "max"])
print("\n--- Medidas de Dispersión ---")
print(dispersion)

# ---------- Posición (cuartiles) ----------
position = df[cols_num].describe(percentiles=[0.25, 0.5, 0.75])
print("\n--- Medidas de Posición (Describe) ---")
print(position)


# ============================================================
# 4) OUTLIERS (IQR) - eliminar si es necesario
#    - Conserva NaN
#    - Evita vaciar columnas con pocos datos no nulos (ej. Gross)
# ============================================================

print("\n============================")
print("4) OUTLIERS (IQR)")
print("============================")

def remove_outliers_iqr(data, column, min_non_null=100):
    non_null = data[column].dropna()

    if non_null.shape[0] < min_non_null:
        print(f"No se eliminan outliers en {column}: pocos datos no nulos ({non_null.shape[0]}).")
        return data

    Q1 = non_null.quantile(0.25)
    Q3 = non_null.quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    before = data.shape[0]
    filtered = data[(data[column].isna()) | ((data[column] >= lower) & (data[column] <= upper))]
    after = filtered.shape[0]

    print(f"{column}: filas antes={before}, después={after} (eliminadas={before-after})")
    return filtered

# Ejemplo típico: VOTES suele tener muchos outliers
df = remove_outliers_iqr(df, "VOTES", min_non_null=100)

# Gross tiene pocos datos (460 aprox), por eso el umbral evita dejarla en 0.
df = remove_outliers_iqr(df, "Gross", min_non_null=100)


# ============================================================
# 5) HISTOGRAMAS (distribución de columnas numéricas)
# ============================================================

print("\n============================")
print("5) HISTOGRAMAS")
print("============================")

df[cols_num].hist(bins=30, figsize=(14, 8))
plt.tight_layout()
plt.show()


# ============================================================
# 6) GRÁFICOS DE DISPERSIÓN (relación entre dos columnas)
# ============================================================

print("\n============================")
print("6) SCATTER PLOTS")
print("============================")

# VOTES vs RATING
plt.figure()
plt.scatter(df["VOTES"], df["RATING"])
plt.xlabel("VOTES")
plt.ylabel("RATING")
plt.title("VOTES vs RATING")
plt.show()

# RunTime vs Gross (solo filas con Gross no nulo)
plt.figure()
sub_gross = df[df["Gross"].notna()]
plt.scatter(sub_gross["RunTime"], sub_gross["Gross"])
plt.xlabel("RunTime")
plt.ylabel("Gross")
plt.title("RunTime vs Gross (solo Gross no nulo)")
plt.show()


# ============================================================
# 7) TRANSFORMACIONES DE COLUMNAS
#    - One Hot Encoding, Label Encoding, Binary Encoding
# ============================================================

print("\n============================")
print("7) TRANSFORMACIONES (ENCODINGS)")
print("============================")

# ONE HOT ENCODING: GENRE
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
genre_encoded = ohe.fit_transform(df[["GENRE"]])

genre_df = pd.DataFrame(
    genre_encoded,
    columns=ohe.get_feature_names_out(["GENRE"])
)

df = pd.concat([df.reset_index(drop=True), genre_df.reset_index(drop=True)], axis=1)

# LABEL ENCODING: MOVIES
le = LabelEncoder()
df["MOVIES_encoded"] = le.fit_transform(df["MOVIES"])

# BINARY ENCODING: YEAR (representación binaria como string)
df["YEAR_binary"] = df["YEAR"].apply(lambda x: format(int(x), "b") if not pd.isna(x) else np.nan)

print("Columnas nuevas creadas:")
print("- OneHot GENRE: ", len(genre_df.columns), "columnas")
print("- MOVIES_encoded")
print("- YEAR_binary")


# ============================================================
# 8) CORRELACIÓN (para decidir si eliminar columnas)
# ============================================================

print("\n============================")
print("8) CORRELACIÓN")
print("============================")

corr_cols = ["RATING", "VOTES", "RunTime", "Gross", "YEAR", "MOVIES_encoded"]
corr_matrix = df[corr_cols].corr(numeric_only=True)

print("\n--- Matriz de correlación ---")
print(corr_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Matriz de Correlación")
plt.show()

# Sugerencia simple de eliminación:
# si dos variables tienen correlación muy alta (|corr| > 0.90), podría considerarse eliminar una.
high_corr_pairs = []
threshold = 0.90

for i in range(len(corr_cols)):
    for j in range(i + 1, len(corr_cols)):
        c = corr_matrix.iloc[i, j]
        if pd.notna(c) and abs(c) >= threshold:
            high_corr_pairs.append((corr_cols[i], corr_cols[j], c))

print("\n--- Pares con alta correlación (|corr| >= 0.90) ---")
if high_corr_pairs:
    for a, b, c in high_corr_pairs:
        print(f"{a} vs {b}: corr={c:.3f}")
else:
    print("No se encontraron pares con correlación extremadamente alta (>= 0.90).")


# ============================================================
# 9) ESCALAMIENTO (MinMax o StandardScaler)
# ============================================================

print("\n============================")
print("9) ESCALAMIENTO")
print("============================")

# Guardar una copia para no perder valores originales (útil para log y reportes)
df_scaled = df.copy()

# Min-Max Scaling (sobre variables numéricas principales)
minmax = MinMaxScaler()
df_scaled[cols_num] = minmax.fit_transform(df_scaled[cols_num])

# StandardScaler (crea columnas adicionales *_std)
std_scaler = StandardScaler()
std_values = std_scaler.fit_transform(df[cols_num])

for idx, c in enumerate(cols_num):
    df_scaled[f"{c}_std"] = std_values[:, idx]

print("Escalamiento aplicado:")
print("- MinMax sobre cols_num (reemplaza columnas en df_scaled)")
print("- StandardScaler en columnas nuevas *_std en df_scaled")


# ============================================================
# 10) TRANSFORMACIÓN LOGARÍTMICA (si es necesario)
#     - Se recomienda aplicarla sobre variables sesgadas (VOTES, Gross)
#     - Se hace ANTES del escalamiento para tener sentido económico
# ============================================================

print("\n============================")
print("10) TRANSFORMACIÓN LOGARÍTMICA")
print("============================")

# Log solo donde hay valores positivos (log1p admite 0, pero no negativos)
# VOTES y Gross suelen ser sesgadas a la derecha
df["VOTES_log"] = np.log1p(df["VOTES"])
df["Gross_log"] = np.log1p(df["Gross"])  # NaN se conserva, log1p(NaN)=NaN

print("Columnas creadas: VOTES_log, Gross_log")


# ============================================================
# 11) CONCLUSIONES (imprimir conclusiones automáticas)
# ============================================================

print("\n============================")
print("11) CONCLUSIONES (AUTOMÁTICAS)")
print("============================")

# 1) Porcentaje de nulos
null_pct = (df[cols_num].isna().mean() * 100).round(2)
print("\n--- % de valores nulos por columna numérica ---")
print(null_pct)

# 2) Sesgo simple (skewness) para detectar necesidad de log
skew_vals = df[cols_num].skew(numeric_only=True).sort_values(ascending=False)
print("\n--- Skewness (asimetría) columnas numéricas ---")
print(skew_vals)

# 3) Hallazgos sobre Gross
gross_count = df["Gross"].notna().sum()
total = df.shape[0]
print("\n--- Hallazgo Gross ---")
print(f"Gross tiene {gross_count} valores no nulos de {total} filas (≈ {(gross_count/total)*100:.2f}%).")
print("Esto limita análisis profundo de Gross; se sugiere:")
print("- No eliminar outliers agresivamente en Gross")
print("- Considerar imputación o trabajar solo con subset donde Gross exista")

# 4) Relación Votes-Rating (correlación)
votes_rating_corr = df[["VOTES", "RATING"]].corr(numeric_only=True).iloc[0, 1]
print("\n--- Relación VOTES vs RATING ---")
print(f"Correlación VOTES-RATING: {votes_rating_corr:.4f} (valores cercanos a 0 indican poca relación lineal).")

# 5) Conclusión final redactada
print("\n--- Conclusión redactada (lista para informe) ---")
print(
    "El conjunto de datos presenta valores faltantes relevantes, especialmente en Gross, "
    "lo cual sugiere que no todas las películas tienen información de recaudo. "
    "Variables como VOTES (y Gross cuando existe) tienden a ser asimétricas a la derecha, "
    "por lo que una transformación logarítmica (log1p) es adecuada para estabilizar su distribución. "
    "A nivel de relación entre variables, la correlación entre VOTES y RATING puede ser baja o moderada, "
    "lo que indica que la popularidad (votos) no necesariamente implica una calificación alta. "
    "Tras limpieza, análisis gráfico, eliminación controlada de outliers y transformaciones (encodings, correlación y escalamiento), "
    "se obtiene un dataset más consistente y preparado para modelado de aprendizaje automático."
)

print("\n============================")
print("FIN DEL SCRIPT")
print("============================")