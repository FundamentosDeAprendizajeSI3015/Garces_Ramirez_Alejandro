"""
Point 3 - Label validation and correction via unsupervised ensemble.

The rubric states that roughly 30% of the labels in the training set
may be incorrect. This module uses the clustering results from Point 2
to flag suspicious students and produce a corrected version of the
training labels that Points 4 and 5 will consume.

The strategy is a three-signal ensemble inspired by the FIRE-UdeA
label validation analysis developed earlier in the course. Each signal
is independent and captures a different failure mode:

    Signal 1 - K-Means discord
        For every K-Means cluster we compute the majority true label.
        A student whose own label disagrees with the majority of its
        cluster is "discordant": other students with a similar profile
        were assigned a different outcome.

    Signal 2 - Fuzzy C-Means ambiguity
        We look at the maximum membership of each student across the
        FCM clusters. If this maximum is below a threshold (0.5), no
        cluster claims the student strongly -> the student lives in a
        fuzzy frontier and its label is uncertain.

    Signal 3 - DBSCAN noise
        DBSCAN marked some students as noise points (-1). These are
        students that do not fit any dense region of the feature
        space, geometrically atypical, and worth a second look.

Every flag contributes one point to a suspicion score in {0, 1, 2, 3}.
Students with score >= 2 are considered "highly suspicious" and their
label is replaced by the majority label of their K-Means cluster.
Students with score <= 1 keep their original label.

The output is a corrected dataset that preserves the original Target
column for comparison and adds a Target_corrected column with the
result of the relabeling procedure.

Artifacts produced:
    - src/outputs/data_corrected.csv    : full dataset + corrected label
    - src/outputs/suspicion_report.csv  : per-student flags and score
    - src/outputs/relabel_summary.json  : aggregate statistics

Run:
    python src/03_relabel.py
"""

# %%
# ============================================================================
# 1. IMPORTS AND CONFIGURATION
# ============================================================================
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import (
    CLASS_ORDER,
    OUTPUTS_DIR,
    TARGET_COL,
    load_raw_data,
    save_figure,
)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11

CLASS_PALETTE = {
    "Dropout": "#d62728",
    "Enrolled": "#ff7f0e",
    "Graduate": "#2ca02c",
}

# Threshold below which a FCM maximum membership is considered too
# low to trust the assignment. 0.5 is the natural choice in a 3-class
# problem: it means no cluster has a majority claim on the student.
FCM_AMBIGUITY_THRESHOLD = 0.5

# Minimum suspicion score to trigger relabeling. Score of 2 means at
# least two of the three independent signals agree that the label is
# suspicious, which is a stronger criterion than any single signal.
SUSPICION_CUTOFF = 2

print("=" * 72)
print("POINT 3 - LABEL VALIDATION AND CORRECTION")
print("=" * 72)


# %%
# ============================================================================
# 2. LOAD RAW DATA AND POINT 2 ARTIFACTS
# ============================================================================
# The clustering artifacts must exist before this module runs. If
# they are missing, the user forgot to run 02_unsupervised.py first.
cluster_labels_path = OUTPUTS_DIR / "cluster_labels.csv"
fcm_memberships_path = OUTPUTS_DIR / "fcm_memberships.npy"
pca_coords_path = OUTPUTS_DIR / "pca_coords.npy"

for path in [cluster_labels_path, fcm_memberships_path, pca_coords_path]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing artifact: {path}\n"
            "Run src/02_unsupervised.py before this module."
        )

df = load_raw_data()
cluster_df = pd.read_csv(cluster_labels_path)
fcm_memberships = np.load(fcm_memberships_path)
pca_coords = np.load(pca_coords_path)

# Sanity check: the clustering file must have the same number of rows
# as the raw data and the labels in cluster_df must match the raw df.
assert len(cluster_df) == len(df), "cluster_labels.csv row count mismatch"
assert (cluster_df["y_true"].values == df[TARGET_COL].values).all(), \
    "cluster_labels.csv target does not match raw data"

print(f"\n[Loaded] {len(df)} students, "
      f"clusterings from {cluster_df.shape[1] - 2} methods, "
      f"FCM memberships shape={fcm_memberships.shape}")

y_true_str = df[TARGET_COL].values
kmeans_labels = cluster_df["KMeans"].values
dbscan_labels = cluster_df["DBSCAN"].values


# %%
# ============================================================================
# 3. SIGNAL 1 - K-MEANS DISCORD
# ============================================================================
# For each K-Means cluster we compute its majority true label. A
# student is discordant iff its own true label differs from the
# majority of its cluster. Intuition: "the average student with a
# profile like yours was classified differently".
unique_clusters = sorted(set(kmeans_labels))
cluster_majority = {}
for c in unique_clusters:
    mask = kmeans_labels == c
    majority = pd.Series(y_true_str[mask]).mode().iloc[0]
    cluster_majority[c] = majority

km_expected_label = np.array([cluster_majority[c] for c in kmeans_labels])
flag_discord = (y_true_str != km_expected_label).astype(int)

print("\n[Signal 1 - K-Means discord]")
print(f"  Cluster majorities : {cluster_majority}")
print(f"  Discordant students: {flag_discord.sum()} "
      f"({flag_discord.mean():.1%})")


# %%
# ============================================================================
# 4. SIGNAL 2 - FCM MAX-MEMBERSHIP AMBIGUITY
# ============================================================================
# Row-wise maximum of the FCM membership matrix. If the max is below
# FCM_AMBIGUITY_THRESHOLD, no cluster claims the student strongly.
max_membership = fcm_memberships.max(axis=1)
flag_ambiguous = (max_membership < FCM_AMBIGUITY_THRESHOLD).astype(int)

print("\n[Signal 2 - FCM ambiguity]")
print(f"  Max membership distribution: "
      f"min={max_membership.min():.3f}, "
      f"mean={max_membership.mean():.3f}, "
      f"max={max_membership.max():.3f}")
print(f"  Ambiguous students (max < {FCM_AMBIGUITY_THRESHOLD}): "
      f"{flag_ambiguous.sum()} ({flag_ambiguous.mean():.1%})")

# --- Figure: histogram of max FCM memberships ---
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(max_membership, bins=40, color="steelblue",
        edgecolor="black", alpha=0.8)
ax.axvline(FCM_AMBIGUITY_THRESHOLD, color="red", linestyle="--",
           linewidth=2, label=f"Ambiguity threshold = {FCM_AMBIGUITY_THRESHOLD}")
ax.set_xlabel("Maximum FCM membership per student")
ax.set_ylabel("Number of students")
ax.set_title("FCM ambiguity: distribution of per-student max membership\n"
             "(left of the red line = ambiguous)")
ax.legend()
save_figure(fig, "07_fcm_ambiguity_histogram")


# %%
# ============================================================================
# 5. SIGNAL 3 - DBSCAN NOISE
# ============================================================================
# DBSCAN noise points (-1) are students that do not belong to any
# dense neighborhood -> geometric outliers worth inspecting.
flag_noise = (dbscan_labels == -1).astype(int)

print("\n[Signal 3 - DBSCAN noise]")
print(f"  Noise students: {flag_noise.sum()} ({flag_noise.mean():.1%})")


# %%
# ============================================================================
# 6. AGGREGATE SUSPICION SCORE
# ============================================================================
# Three independent signals -> score in {0, 1, 2, 3}. We will relabel
# only students with score >= SUSPICION_CUTOFF (default 2).
suspicion_score = flag_discord + flag_ambiguous + flag_noise

score_distribution = pd.Series(suspicion_score).value_counts().sort_index()
print("\n[Aggregate suspicion score]")
for s, count in score_distribution.items():
    pct = count / len(suspicion_score) * 100
    print(f"  Score {s}: {count:>5} students ({pct:5.2f}%)")

# --- Figure: suspicion score distribution ---
fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#2ca02c", "#ffdd66", "#ff9933", "#d62728"]
bars = ax.bar(
    [str(s) for s in score_distribution.index],
    score_distribution.values,
    color=[colors[s] for s in score_distribution.index],
    edgecolor="black",
)
for bar, val in zip(bars, score_distribution.values):
    pct = val / len(suspicion_score) * 100
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            f"{val}\n({pct:.1f}%)",
            ha="center", fontsize=10)
ax.set_xlabel("Suspicion score (number of flags triggered)")
ax.set_ylabel("Number of students")
ax.set_title("Distribution of aggregate suspicion score\n"
             f"(students with score >= {SUSPICION_CUTOFF} will be relabeled)")
ax.set_ylim(0, score_distribution.max() * 1.18)
save_figure(fig, "08_suspicion_score_distribution")


# %%
# ============================================================================
# 7. RELABELING
# ============================================================================
# Students with score >= cutoff get reassigned to the majority label
# of their K-Means cluster. Students with score below cutoff keep
# their original label untouched.
is_suspect = suspicion_score >= SUSPICION_CUTOFF
y_corrected = y_true_str.copy()
y_corrected[is_suspect] = km_expected_label[is_suspect]

n_changed = int((y_corrected != y_true_str).sum())
n_kept = len(y_corrected) - n_changed
print(f"\n[Relabeling] cutoff={SUSPICION_CUTOFF}  "
      f"changed={n_changed} ({n_changed / len(y_corrected):.1%})  "
      f"unchanged={n_kept}")

# Breakdown: from each original class, how many went to each new class.
transition = pd.crosstab(
    pd.Series(y_true_str, name="Original"),
    pd.Series(y_corrected, name="Corrected"),
).reindex(index=CLASS_ORDER, columns=CLASS_ORDER, fill_value=0)
print("\n[Transition matrix - original vs corrected label]")
print(transition.to_string())


# %%
# ============================================================================
# 8. DIAGNOSTIC FIGURES
# ============================================================================
# 8.a Transition heatmap
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    transition, annot=True, fmt="d", cmap="Blues",
    cbar_kws={"label": "Students"}, ax=ax,
)
ax.set_title("Label transitions: original vs corrected")
ax.set_xlabel("Corrected label")
ax.set_ylabel("Original label")
save_figure(fig, "09_relabel_transition_matrix")

# 8.b PCA 2D highlighting suspects vs kept
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: confirmed (unchanged) students colored by their label
ax = axes[0]
kept_mask = ~is_suspect
for cls in CLASS_ORDER:
    m = kept_mask & (y_true_str == cls)
    ax.scatter(pca_coords[m, 0], pca_coords[m, 1],
               c=CLASS_PALETTE[cls], s=8, alpha=0.6,
               label=f"{cls} (n={m.sum()})")
ax.set_title(f"Confirmed students (score < {SUSPICION_CUTOFF})")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend(loc="best", fontsize=9)

# Right: suspects highlighted in black, context in grey
ax = axes[1]
ax.scatter(pca_coords[kept_mask, 0], pca_coords[kept_mask, 1],
           c="lightgrey", s=8, alpha=0.5, label="Confirmed")
ax.scatter(pca_coords[is_suspect, 0], pca_coords[is_suspect, 1],
           c="black", s=14, alpha=0.85,
           label=f"Suspect (n={is_suspect.sum()})")
ax.set_title(f"Suspicious students (score >= {SUSPICION_CUTOFF})")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend(loc="best", fontsize=9)

fig.suptitle("Suspect identification on PCA 2D projection", fontsize=13, y=1.02)
save_figure(fig, "10_suspects_pca_projection")


# %%
# ============================================================================
# 9. SAVE CORRECTED DATASET AND REPORTS
# ============================================================================
# The corrected dataset keeps every original column and adds the
# corrected target. Downstream modules decide which column to use.
df_corrected = df.copy()
df_corrected["Target_corrected"] = y_corrected
df_corrected.to_csv(OUTPUTS_DIR / "data_corrected.csv", index=False)

# Per-student suspicion report: useful for the presentation and for
# any qualitative inspection.
suspicion_report = pd.DataFrame({
    "flag_discord":      flag_discord,
    "flag_ambiguous":    flag_ambiguous,
    "flag_noise":        flag_noise,
    "suspicion_score":   suspicion_score,
    "max_fcm_membership": max_membership.round(4),
    "kmeans_cluster":    kmeans_labels,
    "kmeans_majority":   km_expected_label,
    "original_label":    y_true_str,
    "corrected_label":   y_corrected,
    "was_relabeled":     is_suspect.astype(int),
})
suspicion_report.to_csv(OUTPUTS_DIR / "suspicion_report.csv", index=False)

# Aggregate summary (consumed by Point 5 for the comparison table).
summary = {
    "total_students":          int(len(df)),
    "fcm_ambiguity_threshold": FCM_AMBIGUITY_THRESHOLD,
    "suspicion_cutoff":        SUSPICION_CUTOFF,
    "signal_counts": {
        "discord":   int(flag_discord.sum()),
        "ambiguous": int(flag_ambiguous.sum()),
        "noise":     int(flag_noise.sum()),
    },
    "score_distribution": {
        int(k): int(v) for k, v in score_distribution.items()
    },
    "relabeled_total":         n_changed,
    "relabeled_fraction":      round(n_changed / len(df), 4),
    "transition_matrix":       transition.to_dict(),
}
with open(OUTPUTS_DIR / "relabel_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n[Artifacts saved]")
print(f"  {OUTPUTS_DIR / 'data_corrected.csv'}")
print(f"  {OUTPUTS_DIR / 'suspicion_report.csv'}")
print(f"  {OUTPUTS_DIR / 'relabel_summary.json'}")


# %%
# ============================================================================
# 10. SUMMARY
# ============================================================================
orig_counts = pd.Series(y_true_str).value_counts().reindex(CLASS_ORDER)
corr_counts = pd.Series(y_corrected).value_counts().reindex(CLASS_ORDER)
print("\n" + "=" * 72)
print("RELABELING SUMMARY")
print("=" * 72)
print(f"- Flagged by discord   : {flag_discord.sum():>5} "
      f"({flag_discord.mean():.1%})")
print(f"- Flagged by ambiguity : {flag_ambiguous.sum():>5} "
      f"({flag_ambiguous.mean():.1%})")
print(f"- Flagged by noise     : {flag_noise.sum():>5} "
      f"({flag_noise.mean():.1%})")
print(f"- Students relabeled   : {n_changed:>5} "
      f"({n_changed / len(df):.1%})")
print("\n[Class distribution before -> after]")
for cls in CLASS_ORDER:
    delta = int(corr_counts[cls] - orig_counts[cls])
    sign = "+" if delta >= 0 else ""
    print(f"  {cls:<10} {orig_counts[cls]:>5}  ->  "
          f"{corr_counts[cls]:>5}  ({sign}{delta})")
print("=" * 72)