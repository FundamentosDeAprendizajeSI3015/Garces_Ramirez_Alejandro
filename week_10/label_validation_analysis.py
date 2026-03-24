# ==========================================================
# Clustering Pipeline FIRE-UdeA — Validación de Etiquetas
# Universidad de Antioquia - FIRE-UdeA | Marzo 2026
# ==========================================================
#
# REGLA: 'label' se extrae PRIMERO y se elimina del DataFrame.
#         El clustering nunca la ve. Solo vuelve al final
#         para validación externa y diagnóstico.
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import os, warnings

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
# UMAP — safe import
HAS_UMAP = False
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    print("[WARN] UMAP no disponible — instalar: pip install umap-learn")
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score,
    normalized_mutual_info_score, confusion_matrix,
)
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

# 3D — safe import
HAS_3D = False
try:
    from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
    HAS_3D = True
except Exception:
    pass

# HDBSCAN — safe import
HAS_HDBSCAN = False
try:
    from sklearn.cluster import HDBSCAN
    HAS_HDBSCAN = True
except ImportError:
    try:
        import hdbscan as _h
        HDBSCAN = _h.HDBSCAN
        HAS_HDBSCAN = True
    except ImportError:
        pass

warnings.filterwarnings("ignore")
sns.set(style="whitegrid", context="talk")

OUTPUT_DIR = "graficas_clustering"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Paleta
C0, C1   = "#3498db", "#e74c3c"
C_SUSP   = "#f39c12"
C_CORE   = "#2ecc71"
C_OUT    = "#9b59b6"
PAL_CL   = {0: "#1abc9c", 1: "#e67e22", 2: "#a29bfe", -1: "#b2bec3"}


def _save(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  [fig] {path}")


# ==========================================================
# 1. CARGA Y PREPROCESAMIENTO
# ==========================================================

def load_and_prep(csv_path, exclude_cols=None):
    """
    Carga CSV, separa label, imputa NaN, estandariza.
    Añade features polinómicas de grado 2 (interacciones)
    para aumentar la dimensionalidad antes del clustering.
    Retorna X_scaled, y_true, feature_names
    """
    if exclude_cols is None:
        exclude_cols = []

    df = pd.read_csv(csv_path)
    print(f"\n[CARGA] {csv_path}: {df.shape[0]} filas × {df.shape[1]} cols")

    # ── Separar etiqueta PRIMERO ────────────────────────────────────────────
    y_true    = df["label"].values
    drop_cols = set(["label"] + exclude_cols)
    feat_cols = [c for c in df.columns
                 if c not in drop_cols and df[c].dtype != object]

    print(f"  y_true separado → dist={dict(pd.Series(y_true).value_counts().sort_index())}")
    print(f"  Descartadas: {list(drop_cols)}")
    print(f"  Features base ({len(feat_cols)}): {feat_cols}")

    X = df[feat_cols].copy()
    for col in feat_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())

    # ── Aumentar dimensionalidad: interacciones polinómicas grado 2 ─────────
    poly       = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly     = poly.fit_transform(X.values)
    poly_names = poly.get_feature_names_out(feat_cols)
    print(f"  Tras PolynomialFeatures grado-2: {X_poly.shape[1]} features")

    X_scaled = StandardScaler().fit_transform(X_poly)
    return X_scaled, y_true, list(poly_names)


# ==========================================================
# 2. CLUSTERING
# ==========================================================

def run_all_clustering(X, dbscan_eps, dbscan_min, hdbscan_min, k=2):
    labels = {}
    labels["K-Means"]    = KMeans(n_clusters=k, random_state=42, n_init=15).fit_predict(X)
    labels["Ward"]       = fcluster(linkage(X, method="ward"), t=k, criterion="maxclust") - 1
    labels["DBSCAN"]     = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min).fit_predict(X)
    if HAS_HDBSCAN:
        labels["HDBSCAN"] = HDBSCAN(min_cluster_size=hdbscan_min).fit_predict(X)
    else:
        labels["HDBSCAN"] = labels["K-Means"].copy()
        print("  [INFO] HDBSCAN no disponible — usando K-Means como proxy")
    return labels


# ==========================================================
# 3. DIAGNÓSTICO DE ETIQUETAS
# ==========================================================

def diagnose_labels(X, y_true, km_labels, db_labels, dataset_name):
    """
    Detecta puntos potencialmente mal etiquetados con 3 criterios:
      1. Discordancia K-Means (etiqueta ≠ mayoría del cluster)
      2. Outlier LOF (Local Outlier Factor > umbral)
      3. Ruido DBSCAN (marcado como -1)
    Un punto es SOSPECHOSO si cumple ≥ 2 criterios.
    """
    n = len(X)

    # Criterio 1: discordancia K-Means
    cluster_majority = {c: pd.Series(y_true[km_labels == c]).mode()[0]
                        for c in np.unique(km_labels)}
    km_majority = np.array([cluster_majority[c] for c in km_labels])
    flag_discord = (y_true != km_majority).astype(int)

    # Criterio 2: outlier LOF
    lof      = LocalOutlierFactor(n_neighbors=min(20, n - 1), contamination=0.1)
    lof_pred = lof.fit_predict(X)
    lof_score = -lof.negative_outlier_factor_
    flag_lof  = (lof_pred == -1).astype(int)

    # Criterio 3: ruido DBSCAN
    flag_noise = (db_labels == -1).astype(int)

    suspicion     = flag_discord + flag_lof + flag_noise
    is_suspicious = (suspicion >= 2).astype(int)

    diag = pd.DataFrame({
        "y_true":         y_true,
        "km_cluster":     km_labels,
        "km_majority":    km_majority,
        "lof_score":      lof_score,
        "flag_discord":   flag_discord,
        "flag_lof":       flag_lof,
        "flag_noise":     flag_noise,
        "suspicion":      suspicion,
        "is_suspicious":  is_suspicious,
    })

    # ── Imprimir diagnóstico ────────────────────────────────────────────────
    SEP = "=" * 58
    print(f"\n{SEP}")
    print(f"  DIAGNÓSTICO DE ETIQUETAS — {dataset_name}")
    print(SEP)

    for lv in sorted(np.unique(y_true)):
        m   = diag["y_true"] == lv
        n_l = m.sum()
        print(f"\n  ── Etiqueta = {lv}  (n={n_l}) ──────────────────")
        for flag, desc in [("flag_discord", "Discordancia K-Means"),
                            ("flag_lof",    "Outlier LOF        "),
                            ("flag_noise",  "Ruido DBSCAN       ")]:
            cnt = diag.loc[m, flag].sum()
            print(f"    {desc}: {cnt:3d}  ({100*cnt/n_l:5.1f}%)")
        n_s = diag.loc[m, "is_suspicious"].sum()
        print(f"    SOSPECHOSOS (≥2)   : {n_s:3d}  ({100*n_s/n_l:5.1f}%)")

    # ── Respuesta directa ───────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  ¿CUÁNTOS ETIQUETADOS COMO 1 SON REALMENTE CLASE 1?")
    print(SEP)

    for lv in sorted(np.unique(y_true)):
        m    = diag["y_true"] == lv
        n_l  = m.sum()
        ok   = ((diag["y_true"] == lv) & (diag["km_majority"] == lv) &
                (diag["is_suspicious"] == 0)).sum()
        wrong = ((diag["y_true"] == lv) & (diag["km_majority"] != lv)).sum()
        border = ((diag["y_true"] == lv) & (diag["is_suspicious"] == 1)).sum()
        print(f"\n  label = {lv}  (n={n_l})")
        print(f"    Claramente clase {lv}  : {ok:3d} / {n_l}  =  {100*ok/n_l:5.1f}%")
        print(f"    Posiblemente otra clase: {wrong:3d} / {n_l}  =  {100*wrong/n_l:5.1f}%")
        print(f"    Frontera / dudosos     : {border:3d} / {n_l}  =  {100*border/n_l:5.1f}%")

    # ── Por qué quedaron mal etiquetados ────────────────────────────────────
    n_lof  = diag["flag_lof"].sum()
    n_disc = diag["flag_discord"].sum()
    n_nois = diag["flag_noise"].sum()
    print(f"""
  POR QUÉ QUEDARON MAL ETIQUETADOS:
  ─────────────────────────────────
  • {n_disc} puntos discordantes con K-Means:
    Viven en la región geométrica del cluster contrario.
    Son puntos de FRONTERA donde la señal financiera es ambigua
    y el analista puede haber aplicado criterio subjetivo.

  • {n_lof} puntos detectados como OUTLIERS por LOF:
    Son anómalos en su vecindad local — valores atípicos en
    features financieras (liquidez extrema, CFO muy negativo, etc.)
    que no encajan en ningún cluster denso. El analista los
    clasificó por criterio experto, pero geométricamente "no
    pertenecen" a ninguna clase de forma clara.

  • {n_nois} puntos marcados como RUIDO por DBSCAN:
    No alcanzan la densidad mínima para ser puntos CORE.
    Son puntos NO-CORE que flotan en la frontera entre regiones.
    Exactamente los candidatos a etiquetado inconsistente.

  CONCLUSIÓN: Los puntos mal etiquetados son principalmente
  (1) puntos no-core que viven entre clusters,
  (2) outliers financieros con comportamiento atípico, y
  (3) casos de frontera donde el criterio experto difiere
      de la estructura geométrica de los datos.
""")

    return diag


# ==========================================================
# 4. VISUALIZACIONES RICAS
# ==========================================================

def plot_all_visuals(X, y_true, all_labels, diag, dataset_name):
    """Genera todas las visualizaciones en un pipeline."""

    km  = all_labels["K-Means"]
    db  = all_labels["DBSCAN"]
    susp = diag["is_suspicious"].values == 1

    # PCA 2D y 3D
    pca2_obj = PCA(n_components=2, random_state=42)
    Z2       = pca2_obj.fit_transform(X)
    var2     = pca2_obj.explained_variance_ratio_
    pc1l = f"PC1 ({var2[0]*100:.1f}% var)"
    pc2l = f"PC2 ({var2[1]*100:.1f}% var)"

    pca3_obj = PCA(n_components=3, random_state=42)
    Z3       = pca3_obj.fit_transform(X)

    # UMAP
    if HAS_UMAP:
        print(f"  [UMAP] calculando…")
        n_nb = min(15, len(X) - 1)
        Zu   = UMAP(n_components=2, n_neighbors=n_nb,
                    min_dist=0.1, random_state=42).fit_transform(X)
        umap_label = f"UMAP (n_neighbors={n_nb})"
    else:
        print("  [UMAP] no disponible — panel UMAP omitido")
        Zu = None

    # ── Fig 1: panel maestro 3×2 ────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 18))
    fig.suptitle(f"Análisis de Clustering — {dataset_name}",
                 fontsize=17, fontweight="bold", y=1.01)
    gs = GridSpec(3, 2, figure=fig, hspace=0.40, wspace=0.30)

    def _scatter(ax, Z, labels, title, palette, xlabel="PC1", ylabel="PC2"):
        for lv in sorted(set(labels)):
            m   = np.array(labels) == lv
            col = palette.get(lv, "#999")
            lbl = f"Cluster {lv}" if lv >= 0 else "Ruido"
            ax.scatter(Z[m, 0], Z[m, 1], c=col, s=35, alpha=0.72,
                       linewidths=0, label=lbl)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(fontsize=8, markerscale=1.4, loc="best")

    # [0,0] Labels originales
    ax = fig.add_subplot(gs[0, 0])
    _scatter(ax, Z2, y_true, "Etiquetas originales (PCA 2D)",
             {0: C0, 1: C1}, pc1l, pc2l)

    # [0,1] K-Means clusters + centroides
    ax = fig.add_subplot(gs[0, 1])
    _scatter(ax, Z2, km, "K-Means — Clusters (PCA 2D)", PAL_CL, pc1l, pc2l)
    for c in np.unique(km):
        if c < 0: continue
        cx, cy = Z2[km == c, 0].mean(), Z2[km == c, 1].mean()
        ax.scatter(cx, cy, marker="*", s=400, c="white",
                   edgecolors="black", linewidths=1.2, zorder=10)

    # [1,0] DBSCAN
    ax = fig.add_subplot(gs[1, 0])
    _scatter(ax, Z2, db, "DBSCAN (PCA 2D)  |  -1 = ruido", PAL_CL, pc1l, pc2l)

    # [1,1] Ward jerárquico
    ax = fig.add_subplot(gs[1, 1])
    _scatter(ax, Z2, all_labels["Ward"], "Ward Jerárquico (PCA 2D)", PAL_CL, pc1l, pc2l)

    # [2,0] Sospechosos sobre PCA
    ax = fig.add_subplot(gs[2, 0])
    normal = ~susp
    ax.scatter(Z2[normal, 0], Z2[normal, 1],
               c=[C0 if v == 0 else C1 for v in y_true[normal]],
               s=25, alpha=0.25, linewidths=0)
    ax.scatter(Z2[susp, 0], Z2[susp, 1], c=C_SUSP, s=90, alpha=0.95,
               edgecolors="black", linewidths=0.5, zorder=5)
    ax.set_title("Puntos potencialmente mal etiquetados", fontsize=12, fontweight="bold")
    ax.set_xlabel(pc1l, fontsize=9); ax.set_ylabel(pc2l, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    patches = [mpatches.Patch(color=C0, label="Label 0 (ok)"),
               mpatches.Patch(color=C1, label="Label 1 (ok)"),
               mpatches.Patch(color=C_SUSP, label=f"Sospechoso ({susp.sum()})")]
    ax.legend(handles=patches, fontsize=8)

    # [2,1] UMAP labels originales (o PCA si no hay UMAP)
    ax = fig.add_subplot(gs[2, 1])
    if Zu is not None:
        _scatter(ax, Zu, y_true, f"UMAP — Etiquetas originales",
                 {0: C0, 1: C1}, "UMAP 1", "UMAP 2")
    else:
        _scatter(ax, Z2, km, "K-Means (PCA 2D) — sin UMAP", PAL_CL, pc1l, pc2l)

    fig.tight_layout()
    _save(f"panel_maestro_{dataset_name}.png")

    # ── Fig 2: UMAP comparativo (labels vs clusters vs sospechosos) ─────────
    if Zu is not None:
        fig, axes = plt.subplots(1, 3, figsize=(21, 7))
        fig.suptitle(f"UMAP — {dataset_name}",
                     fontsize=14, fontweight="bold")

        _scatter(axes[0], Zu, y_true, "Etiquetas originales",
                 {0: C0, 1: C1}, "UMAP 1", "UMAP 2")

        _scatter(axes[1], Zu, km, "K-Means clusters  (★ = centroide)",
                 PAL_CL, "UMAP 1", "UMAP 2")
        for c in np.unique(km):
            if c < 0: continue
            cx, cy = Zu[km == c, 0].mean(), Zu[km == c, 1].mean()
            axes[1].scatter(cx, cy, marker="*", s=400, c="white",
                            edgecolors="black", linewidths=1.2, zorder=10)

        # Panel 3: sospechosos con tamaño ∝ LOF score
        lof_s = diag["lof_score"].values
        axes[2].scatter(Zu[~susp, 0], Zu[~susp, 1],
                        c=[C0 if v == 0 else C1 for v in y_true[~susp]],
                        s=20, alpha=0.25, linewidths=0)
        sizes_s = np.clip(lof_s[susp] * 60, 80, 450)
        sc = axes[2].scatter(Zu[susp, 0], Zu[susp, 1],
                             c=lof_s[susp], cmap="YlOrRd",
                             s=sizes_s, alpha=0.95,
                             edgecolors="black", linewidths=0.5, zorder=5)
        plt.colorbar(sc, ax=axes[2], label="LOF Score (↑ = más outlier)")
        axes[2].set_title(f"Sospechosos (n={susp.sum()})  —  tamaño ∝ anomalía",
                          fontsize=12, fontweight="bold")
        axes[2].set_xlabel("UMAP 1"); axes[2].set_ylabel("UMAP 2")
        axes[2].set_xticks([]); axes[2].set_yticks([])

        fig.tight_layout()
        _save(f"umap_comparativo_{dataset_name}.png")

    # ── Fig 3: 3D (nativo o fallback 2D enriquecido) ────────────────────────
    if HAS_3D:
        fig = plt.figure(figsize=(18, 8))
        fig.suptitle(f"PCA 3D — {dataset_name}", fontsize=14, fontweight="bold")

        ax1 = fig.add_subplot(121, projection="3d")
        for lv, col, lbl in [(0, C0, "Label 0"), (1, C1, "Label 1")]:
            m = y_true == lv
            ax1.scatter(Z3[m, 0], Z3[m, 1], Z3[m, 2],
                        c=col, s=25, alpha=0.65, label=lbl)
        ax1.set_title("Etiquetas originales"); ax1.legend(fontsize=9)
        ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2"); ax1.set_zlabel("PC3")

        ax2 = fig.add_subplot(122, projection="3d")
        for c in np.unique(km):
            m   = (km == c) & (~susp)
            col = PAL_CL.get(c, "#999")
            lbl = f"Cluster {c}" if c >= 0 else "Ruido"
            ax2.scatter(Z3[m, 0], Z3[m, 1], Z3[m, 2],
                        c=col, s=20, alpha=0.50, label=lbl)
        ax2.scatter(Z3[susp, 0], Z3[susp, 1], Z3[susp, 2],
                    c=C_SUSP, s=100, alpha=1.0, marker="^",
                    edgecolors="black", linewidths=0.5,
                    zorder=10, label=f"Sospechoso ({susp.sum()})")
        ax2.set_title("Clusters + sospechosos ▲"); ax2.legend(fontsize=8)
        ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2"); ax2.set_zlabel("PC3")

        fig.tight_layout()
        _save(f"pca3d_{dataset_name}.png")
    else:
        # Fallback: PC1 vs PC3 coloreado por PC2
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(f"Pseudo-3D (PC1×PC2×PC3 cod. tamaño) — {dataset_name}",
                     fontsize=13, fontweight="bold")

        size_pc3 = 20 + 80 * (Z3[:, 2] - Z3[:, 2].min()) / (np.ptp(Z3[:, 2]) + 1e-9)
        for lv, col, lbl in [(0, C0, "Label 0"), (1, C1, "Label 1")]:
            m = y_true == lv
            axes[0].scatter(Z3[m, 0], Z3[m, 1], c=col,
                            s=size_pc3[m], alpha=0.65, linewidths=0, label=lbl)
        axes[0].set_title("Etiquetas | tamaño ∝ PC3"); axes[0].legend(fontsize=9)
        axes[0].set_xlabel(pc1l); axes[0].set_ylabel(pc2l)

        for c in np.unique(km):
            m   = (km == c) & (~susp)
            col = PAL_CL.get(c, "#999")
            axes[1].scatter(Z3[m, 0], Z3[m, 1], c=col,
                            s=size_pc3[m], alpha=0.50, linewidths=0,
                            label=f"Cluster {c}" if c >= 0 else "Ruido")
        axes[1].scatter(Z3[susp, 0], Z3[susp, 1], c=C_SUSP,
                        s=120, alpha=0.95, marker="^",
                        edgecolors="black", linewidths=0.5,
                        label=f"Sospechoso ({susp.sum()})")
        axes[1].set_title("Clusters + sospechosos | tamaño ∝ PC3"); axes[1].legend(fontsize=8)
        axes[1].set_xlabel(pc1l); axes[1].set_ylabel(pc2l)

        fig.tight_layout()
        _save(f"pseudo3d_{dataset_name}.png")

    # ── Fig 4: LOF por clase ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"LOF Score por clase — {dataset_name}", fontsize=13, fontweight="bold")

    for lv, col in [(0, C0), (1, C1)]:
        vals = diag.loc[diag["y_true"] == lv, "lof_score"]
        axes[0].hist(vals, bins=25, color=col, alpha=0.55,
                     label=f"Label {lv}  (μ={vals.mean():.2f})", edgecolor="white")
    axes[0].axvline(1.5, color="black", linestyle="--", lw=1.2, label="Umbral ~1.5")
    axes[0].set_xlabel("LOF Score"); axes[0].set_ylabel("Frecuencia")
    axes[0].set_title("Histograma LOF por clase"); axes[0].legend(fontsize=9)

    data = [diag.loc[diag["y_true"] == lv, "lof_score"].values
            for lv in sorted(np.unique(y_true))]
    bp = axes[1].boxplot(data, patch_artist=True,
                         tick_labels=[f"Label {lv}" for lv in sorted(np.unique(y_true))])
    for patch, col in zip(bp["boxes"], [C0, C1]):
        patch.set_facecolor(col); patch.set_alpha(0.6)
    axes[1].axhline(1.5, color="black", linestyle="--", lw=1.2)
    axes[1].set_ylabel("LOF Score"); axes[1].set_title("Boxplot LOF por clase")
    fig.tight_layout()
    _save(f"lof_por_clase_{dataset_name}.png")

    # ── Fig 5: heatmap P(clase|cluster) para K-Means ────────────────────────
    clusters = sorted(np.unique(km))
    classes  = sorted(np.unique(y_true))
    mat = np.array([[((y_true == cl) & (km == c)).sum() / (km == c).sum()
                     for cl in classes] for c in clusters])
    fig, ax = plt.subplots(figsize=(7, max(3, len(clusters) * 1.2)))
    sns.heatmap(mat, annot=True, fmt=".3f", cmap="RdYlGn",
                xticklabels=[f"clase_{c}" for c in classes],
                yticklabels=[f"Cluster {c}" for c in clusters],
                vmin=0, vmax=1, linewidths=0.5, ax=ax)
    ax.set_title(f"P(clase | cluster) — K-Means — {dataset_name}", fontsize=12)
    ax.set_ylabel("Cluster"); ax.set_xlabel("Clase")
    plt.tight_layout()
    _save(f"prob_condicional_{dataset_name}.png")

    # ── Fig 6: Matriz de confusión (etiqueta real vs cluster K-Means) ────────
    # Alineamos clusters con clases usando el mapeo de mayoría
    cluster_majority = {c: pd.Series(y_true[km == c]).mode()[0]
                        for c in np.unique(km) if c >= 0}
    km_mapped = np.array([cluster_majority.get(c, -1) for c in km])
    valid     = km_mapped >= 0
    cm_mat    = confusion_matrix(y_true[valid], km_mapped[valid],
                                  labels=sorted(np.unique(y_true)))
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_mat, annot=True, fmt="d", cmap="Blues",
                xticklabels=[f"Predicho\nlabel={c}" for c in sorted(np.unique(y_true))],
                yticklabels=[f"Real\nlabel={c}" for c in sorted(np.unique(y_true))],
                linewidths=1, ax=ax)
    ax.set_title(f"Matriz de Confusión — K-Means vs Etiquetas Reales\n{dataset_name}",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Etiqueta Real"); ax.set_xlabel("Cluster K-Means (mapeado por mayoría)")
    plt.tight_layout()
    _save(f"confusion_matrix_{dataset_name}.png")

    # ── Fig 7: Diagrama de pastel — Bien vs Mal etiquetados ──────────────────
    n_total   = len(y_true)
    n_ok      = int((~susp).sum())
    n_border  = int(susp.sum())
    # Subdividir sospechosos por tipo dominante
    n_discord = int(diag.loc[susp, "flag_discord"].sum())
    n_outlier = int(diag.loc[susp, "flag_lof"].sum())
    n_noise   = int(diag.loc[susp, "flag_noise"].sum())

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"Diagnóstico de Etiquetas — {dataset_name}",
                 fontsize=14, fontweight="bold")

    # Pastel izquierdo: global bien vs sospechoso
    sizes_g  = [n_ok, n_border]
    labels_g = [f"Bien etiquetados\n({n_ok} pts, {100*n_ok/n_total:.1f}%)",
                f"Sospechosos\n({n_border} pts, {100*n_border/n_total:.1f}%)"]
    colors_g = [C_CORE, C_SUSP]
    explode  = (0, 0.08)
    wedges, texts, autotexts = axes[0].pie(
        sizes_g, labels=labels_g, colors=colors_g, explode=explode,
        autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 11},
        wedgeprops={"edgecolor": "white", "linewidth": 2})
    for at in autotexts:
        at.set_fontsize(12); at.set_fontweight("bold")
    axes[0].set_title("¿Bien o mal etiquetados?\n(global)", fontsize=13, fontweight="bold")

    # Pastel derecho: desglose de los sospechosos por motivo
    # Evitar pastel vacío si no hay sospechosos
    if n_border > 0:
        # Contar cuántos tienen SOLO cada flag como dominante
        d_sub = diag[susp].copy()
        solo_discord = ((d_sub["flag_discord"] == 1) & (d_sub["flag_lof"] == 0) & (d_sub["flag_noise"] == 0)).sum()
        solo_outlier = ((d_sub["flag_discord"] == 0) & (d_sub["flag_lof"] == 1) & (d_sub["flag_noise"] == 0)).sum()
        solo_noise   = ((d_sub["flag_discord"] == 0) & (d_sub["flag_lof"] == 0) & (d_sub["flag_noise"] == 1)).sum()
        combinados   = n_border - solo_discord - solo_outlier - solo_noise

        sizes_s  = [solo_discord, solo_outlier, solo_noise, combinados]
        labels_s = [f"Solo discordante\nK-Means ({solo_discord})",
                    f"Solo outlier\nLOF ({solo_outlier})",
                    f"Solo ruido\nDBSCAN ({solo_noise})",
                    f"Combinados\n({combinados})"]
        colors_s = ["#e67e22", C_OUT, "#e74c3c", "#636e72"]
        # Filtrar categorías vacías
        nz = [(s, l, c) for s, l, c in zip(sizes_s, labels_s, colors_s) if s > 0]
        if nz:
            sizes_s, labels_s, colors_s = zip(*nz)
            axes[1].pie(list(sizes_s), labels=list(labels_s), colors=list(colors_s),
                        autopct="%1.1f%%", startangle=90,
                        textprops={"fontsize": 10},
                        wedgeprops={"edgecolor": "white", "linewidth": 2})
        axes[1].set_title(f"¿Por qué son sospechosos?\n(n={n_border})",
                          fontsize=13, fontweight="bold")
    else:
        axes[1].text(0.5, 0.5, "Sin puntos sospechosos\n¡Etiquetado perfecto!",
                     ha="center", va="center", fontsize=14, color=C_CORE,
                     fontweight="bold", transform=axes[1].transAxes)
        axes[1].set_title("Desglose de sospechosos", fontsize=13, fontweight="bold")

    plt.tight_layout()
    _save(f"pastel_diagnostico_{dataset_name}.png")

    print(f"  [OK] Todas las figuras de {dataset_name} generadas.")


# ==========================================================
# 5. VEREDICTO SOBRE LOS ANALISTAS
# ==========================================================

def veredicto_analistas(diag_d1, diag_d2):
    SEP = "=" * 58

    pct1 = 100 * diag_d1["is_suspicious"].mean()
    pct2 = 100 * diag_d2["is_suspicious"].mean()
    avg  = (pct1 + pct2) / 2

    # Discordancia en label=1 (el caso más revelador)
    def _discord1(d):
        m = d["y_true"] == 1
        return 100 * d.loc[m, "flag_discord"].mean() if m.sum() > 0 else 0.0

    disc1 = _discord1(diag_d1)
    disc2 = _discord1(diag_d2)

    print(f"\n{SEP}")
    print("  VEREDICTO SOBRE LOS ANALISTAS FINANCIEROS")
    print(SEP)
    print(f"""
  Dataset 1 (sintético, n=500):
    Puntos sospechosos   : {pct1:.1f}%
    Discordancia label=1 : {disc1:.1f}%

  Dataset 2 (realista,  n=80):
    Puntos sospechosos   : {pct2:.1f}%
    Discordancia label=1 : {disc2:.1f}%

  Promedio global de sospecha : {avg:.1f}%
""")

    if avg < 10:
        decision = "ASCENDER A LOS ANALISTAS ★ ★ ★"
        razon = (
            f"Solo el {avg:.1f}% de los puntos presenta señales de etiquetado\n"
            "  inconsistente. La taxonomía financiera tiene alta coherencia\n"
            "  geométrica. Las discordancias restantes corresponden a casos\n"
            "  genuinamente ambiguos en la frontera entre clases, donde\n"
            "  cualquier experto dudaría. El criterio es sólido."
        )
    elif avg < 25:
        decision = "CAPACITAR ANTES DE DECIDIR ⚠ ⚠ ⚠"
        razon = (
            f"{avg:.1f}% de sospecha global. Los analistas clasifican bien los\n"
            "  casos claros pero son inconsistentes en zonas de alta densidad\n"
            "  mixta. Se recomienda revisión del protocolo de etiquetado y\n"
            "  re-anotación de los puntos sospechosos identificados antes\n"
            "  de tomar decisiones de personal."
        )
    else:
        decision = "CUESTIONAR EL PROCESO ✗ ✗ ✗"
        razon = (
            f"{avg:.1f}% de sospecha global. El clustering no-supervisado\n"
            "  contradice sistemáticamente más de 1 de cada 4 etiquetas.\n"
            "  El criterio de etiquetado no es consistente con la estructura\n"
            "  intrínseca de los datos. Se recomienda auditoría completa\n"
            "  y redefinición de las clases con criterio cuantitativo."
        )

    print(f"  DECISIÓN: {decision}")
    print(f"\n  Justificación:")
    print(f"  {razon}")
    print(f"\n  Metodología de detección:")
    print(f"  Un punto es SOSPECHOSO si cumple ≥ 2 de:")
    print(f"    (1) Su etiqueta difiere de la mayoría de su cluster K-Means")
    print(f"    (2) LOF > umbral → es outlier en su vecindad local")
    print(f"    (3) DBSCAN lo marca como ruido → no es punto CORE")
    print(f"\n  Gráficos guardados en: {OUTPUT_DIR}/\n")


# ==========================================================
# 6. MAIN
# ==========================================================

if __name__ == "__main__":

    # ── Dataset 1: Sintético ──────────────────────────────────────────────
    X1, y1, _ = load_and_prep(
        "dataset_sintetico_FIRE_UdeA.csv",
        exclude_cols=[],
    )
    km1  = KMeans(n_clusters=2, random_state=42, n_init=15).fit_predict(X1)
    db1  = DBSCAN(eps=4.2, min_samples=10).fit_predict(X1)
    all1 = run_all_clustering(X1, dbscan_eps=4.2, dbscan_min=10,
                              hdbscan_min=15, k=2)
    diag1 = diagnose_labels(X1, y1, km1, db1, "D1_Sintetico")
    plot_all_visuals(X1, y1, all1, diag1, "D1_Sintetico")

    # ── Dataset 2: Realista ───────────────────────────────────────────────
    X2, y2, _ = load_and_prep(
        "dataset_sintetico_FIRE_UdeA_realista.csv",
        exclude_cols=["anio", "unidad"],
    )
    km2  = KMeans(n_clusters=2, random_state=42, n_init=15).fit_predict(X2)
    db2  = DBSCAN(eps=9.5, min_samples=5).fit_predict(X2)
    all2 = run_all_clustering(X2, dbscan_eps=9.5, dbscan_min=5,
                              hdbscan_min=5, k=2)
    diag2 = diagnose_labels(X2, y2, km2, db2, "D2_Realista")
    plot_all_visuals(X2, y2, all2, diag2, "D2_Realista")

    # ── Veredicto final ───────────────────────────────────────────────────
    veredicto_analistas(diag1, diag2)