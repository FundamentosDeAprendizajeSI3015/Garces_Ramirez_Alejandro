"""
Point 1 - Data loading and exploratory data analysis (EDA).

This module covers the first stage of the ML lifecycle for the dropout
prediction project:

    1. Load the UCI dataset (4424 students, 36 features, 3 target classes).
    2. Report basic structural information (shape, dtypes, missing, duplicates).
    3. Analyze the target distribution (class imbalance).
    4. Produce descriptive statistics grouped by feature category.
    5. Generate the core EDA figures used later in the presentation:
         - Target class distribution
         - Boxplots of key academic indicators by target class
         - Correlation heatmap of numerical features vs target

All figures are saved to src/outputs/figures/ via the shared save_figure
helper. Nothing in this module modifies the dataset: its sole purpose is
to understand the data before any preprocessing or modelling.

Run:
    cd early_prediction_university_academic_dropout
    python src/01_load_and_eda.py
"""

# %%
# ============================================================================
# 1. IMPORTS AND CONFIGURATION
# ============================================================================
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend: we only save figures to disk.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import (
    CLASS_ORDER,
    FEATURE_GROUPS,
    TARGET_COL,
    load_raw_data,
    save_figure,
    split_features_target,
)

# Visual defaults used across every figure the pipeline produces.
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11

# Fixed color palette so the Dropout class is always red, Enrolled orange
# and Graduate green across EDA, clustering and supervised results.
CLASS_PALETTE = {
    "Dropout": "#d62728",
    "Enrolled": "#ff7f0e",
    "Graduate": "#2ca02c",
}

print("=" * 72)
print("POINT 1 - DATA LOADING AND EXPLORATORY DATA ANALYSIS")
print("=" * 72)


# %%
# ============================================================================
# 2. LOAD DATA
# ============================================================================
df = load_raw_data()
X, y = split_features_target(df)

print(f"\n[Shape] {df.shape[0]} rows x {df.shape[1]} columns")
print(f"[Features] {X.shape[1]} | [Target] '{TARGET_COL}'")


# %%
# ============================================================================
# 3. STRUCTURAL QUALITY CHECKS
# ============================================================================
# These checks answer the basic "is the data clean?" question. We do
# NOT modify anything here -- any imputation or cleaning will be done
# explicitly and justified in later modules.
n_missing = df.isnull().sum().sum()
n_duplicates = df.duplicated().sum()
n_numeric = df.select_dtypes(include=[np.number]).shape[1]
n_non_numeric = df.shape[1] - n_numeric

print("\n[Quality checks]")
print(f"  Missing values (total cells) : {n_missing}")
print(f"  Duplicated rows              : {n_duplicates}")
print(f"  Numeric columns              : {n_numeric}")
print(f"  Non-numeric columns          : {n_non_numeric}")

# Sanity check: the target is the only non-numeric column in the raw file.
assert n_non_numeric == 1, "Unexpected non-numeric columns beyond the Target."
assert n_missing == 0, "Dataset unexpectedly contains missing values."


# %%
# ============================================================================
# 4. TARGET DISTRIBUTION (CLASS IMBALANCE)
# ============================================================================
# Understanding class imbalance is critical: if the dropout class is
# rare, accuracy becomes a misleading metric and we must prioritise
# recall on the minority class. We report both absolute counts and
# percentages in the canonical class order.
counts = y.value_counts().reindex(CLASS_ORDER)
percentages = (counts / len(y) * 100).round(2)

print("\n[Target distribution]")
for cls in CLASS_ORDER:
    print(f"  {cls:<10} {counts[cls]:>5}  ({percentages[cls]:>5.2f}%)")

# --- Figure: target class distribution ---
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(
    counts.index,
    counts.values,
    color=[CLASS_PALETTE[c] for c in counts.index],
    edgecolor="black",
)
for bar, pct in zip(bars, percentages.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 20,
        f"{pct:.1f}%",
        ha="center",
        fontsize=11,
        fontweight="bold",
    )
ax.set_ylabel("Number of students")
ax.set_title("Target class distribution")
ax.set_ylim(0, counts.max() * 1.15)
save_figure(fig, "01_target_distribution")


# %%
# ============================================================================
# 5. DESCRIPTIVE STATISTICS BY FEATURE GROUP
# ============================================================================
# Instead of dumping describe() over all 36 columns, we summarize by
# conceptual group (academic / demographic / socioeconomic / etc.).
# This matches the grouping defined in README.md and makes the EDA
# directly reusable in the presentation slides.
print("\n[Descriptive statistics by feature group]")
for group_name, group_cols in FEATURE_GROUPS.items():
    group_df = df[group_cols]
    print(f"\n  -- {group_name} ({len(group_cols)} features) --")
    summary = group_df.describe().T[["mean", "std", "min", "max"]]
    print(summary.round(2).to_string())


# %%
# ============================================================================
# 6. KEY ACADEMIC INDICATORS vs TARGET
# ============================================================================
# The literature on dropout consistently points at the number of
# approved curricular units and the average grade of the 1st and 2nd
# semester as the strongest early predictors. We plot them side by
# side as boxplots broken down by target class. The visual gap
# between classes here already anticipates how separable the problem
# is for a classifier.
key_features = [
    "Curricular units 1st sem (approved)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (grade)",
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, feature in zip(axes.flat, key_features):
    sns.boxplot(
        data=df,
        x=TARGET_COL,
        y=feature,
        hue=TARGET_COL,
        order=CLASS_ORDER,
        palette=CLASS_PALETTE,
        legend=False,
        ax=ax,
    )
    ax.set_title(feature, fontsize=10)
    ax.set_xlabel("")
fig.suptitle("Key academic indicators by target class", fontsize=13, y=1.02)
save_figure(fig, "02_key_features_by_target")


# %%
# ============================================================================
# 7. CORRELATION OF NUMERIC FEATURES WITH THE TARGET
# ============================================================================
# For the correlation heatmap we need a numeric target. We encode it
# with the canonical CLASS_ORDER (Dropout=0, Enrolled=1, Graduate=2)
# ONLY for this exploratory plot -- this encoding is not propagated
# to any supervised model, since imposing an ordinal relation on a
# nominal target would be a modelling error. Here it is acceptable
# because a Pearson correlation with this encoding tells us, roughly,
# which features move monotonically from Dropout towards Graduate.
df_encoded = df.copy()
df_encoded[TARGET_COL] = df_encoded[TARGET_COL].map(
    {c: i for i, c in enumerate(CLASS_ORDER)}
)

numeric_df = df_encoded.select_dtypes(include=[np.number])
corr_with_target = (
    numeric_df.corr()[TARGET_COL]
    .drop(TARGET_COL)
    .sort_values(key=np.abs, ascending=False)
)

print("\n[Top 10 features by |correlation with target|]")
print(corr_with_target.head(10).round(3).to_string())

# --- Figure: top correlations with target ---
top_n = 15
top_corr = corr_with_target.head(top_n)
fig, ax = plt.subplots(figsize=(9, 7))
colors = ["#2ca02c" if v > 0 else "#d62728" for v in top_corr.values]
ax.barh(top_corr.index[::-1], top_corr.values[::-1], color=colors[::-1])
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Pearson correlation with encoded target")
ax.set_title(f"Top {top_n} features correlated with target\n"
             "(green = higher values push towards Graduate, "
             "red = towards Dropout)")
save_figure(fig, "03_top_correlations_with_target")


# %%
# ============================================================================
# 8. SUMMARY
# ============================================================================
# A compact text summary is printed at the end so the console output
# can be copy-pasted into the report or the speaker script without
# needing to re-open the figures.
print("\n" + "=" * 72)
print("EDA SUMMARY")
print("=" * 72)
print(f"- Dataset: {df.shape[0]} students x {df.shape[1] - 1} features")
print(f"- Missing values: {n_missing} (dataset is clean)")
print(f"- Duplicated rows: {n_duplicates}")
print(f"- Classes: {', '.join(f'{c}={counts[c]} ({percentages[c]:.1f}%)' for c in CLASS_ORDER)}")
print(f"- Minority class: {counts.idxmin()} -> recall on this class is the priority")
print(f"- Strongest predictor: {corr_with_target.index[0]} "
      f"(|r|={abs(corr_with_target.iloc[0]):.3f})")
print(f"- Figures saved to: src/outputs/figures/")
print("=" * 72)