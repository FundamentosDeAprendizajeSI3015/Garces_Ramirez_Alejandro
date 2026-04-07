"""
Point 4 - Supervised training with original and corrected labels.

This module trains three supervised models (Decision Tree, Logistic
Regression and Linear Regression) on the dropout dataset and evaluates
them under two different training label sources:

    A) Original labels  - the Target column straight from the UCI file.
    B) Corrected labels - the Target_corrected column produced by the
                          unsupervised ensemble in Point 3.

The key methodological point: both model variants are evaluated on the
SAME test split, using the ORIGINAL labels as ground truth. The test
set never sees the corrected labels. This guarantees a fair comparison:
we are asking whether training on the cleaned labels improves
generalisation to the real-world label distribution.

On Linear Regression:
    Linear regression is NOT a classification algorithm, but the rubric
    requires it explicitly. We include it on purpose, encoding the
    target as an ordinal {0, 1, 2} and rounding predictions to the
    closest class. We expect this baseline to underperform both
    proper classifiers -- and the gap is pedagogically valuable: it
    demonstrates, empirically, why the choice of model family matters
    for the nature of the task.

Run:
    python src/04_supervised.py
"""

# %%
# ============================================================================
# 1. IMPORTS AND CONFIGURATION
# ============================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from utils import (
    CLASS_ORDER,
    OUTPUTS_DIR,
    TARGET_COL,
    save_figure,
    split_features_target,
)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11

RANDOM_STATE = 42
TEST_SIZE = 0.25
CLASS_TO_INT = {c: i for i, c in enumerate(CLASS_ORDER)}
INT_TO_CLASS = {i: c for c, i in CLASS_TO_INT.items()}

print("=" * 72)
print("POINT 4 - SUPERVISED TRAINING (ORIGINAL vs CORRECTED LABELS)")
print("=" * 72)


# %%
# ============================================================================
# 2. LOAD CORRECTED DATASET FROM POINT 3
# ============================================================================
# data_corrected.csv carries BOTH label columns:
#   - 'Target'           : original UCI labels
#   - 'Target_corrected' : labels after the Point 3 ensemble correction
# We keep them both in the frame and choose which column to use at
# training time.
corrected_path = OUTPUTS_DIR / "data_corrected.csv"
if not corrected_path.exists():
    raise FileNotFoundError(
        f"Missing {corrected_path}. Run src/03_relabel.py first."
    )

df = pd.read_csv(corrected_path)
assert "Target_corrected" in df.columns, \
    "Corrected dataset does not contain Target_corrected column."

print(f"\n[Loaded] {len(df)} rows, {df.shape[1]} columns "
      f"(includes Target and Target_corrected)")


# %%
# ============================================================================
# 3. STRATIFIED TRAIN / TEST SPLIT
# ============================================================================
# The split is stratified on the ORIGINAL Target so the test set
# preserves the real-world class balance. Applying stratification on
# the corrected target would hide the real distribution from the
# evaluation and bias the comparison in favour of the corrected model.
X_all, y_orig = split_features_target(df[[c for c in df.columns
                                          if c != "Target_corrected"]])
y_corr_full = df["Target_corrected"].values

X_train, X_test, y_train_orig, y_test_orig, y_train_corr, _ = train_test_split(
    X_all.values,
    y_orig.values,
    y_corr_full,
    test_size=TEST_SIZE,
    stratify=y_orig.values,
    random_state=RANDOM_STATE,
)

print(f"\n[Split] train={len(X_train)}  test={len(X_test)}")
print(f"[Train] original class distribution: "
      f"{dict(pd.Series(y_train_orig).value_counts())}")
print(f"[Train] corrected class distribution: "
      f"{dict(pd.Series(y_train_corr).value_counts())}")
print(f"[Test]  class distribution (always original): "
      f"{dict(pd.Series(y_test_orig).value_counts())}")

# Report how many training labels actually differ between the two
# variants. This number, restricted to the training split, is what
# will drive any difference between the two model families.
n_diff = int((y_train_orig != y_train_corr).sum())
print(f"[Train] labels changed by Point 3: {n_diff} "
      f"({n_diff / len(y_train_orig):.1%})")


# %%
# ============================================================================
# 4. SCALING
# ============================================================================
# Decision trees are scale-invariant, but logistic and linear
# regression are not. We fit the scaler on the training split only
# (standard ML hygiene: no information from the test split may leak
# into the fitting of any preprocessing step).
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)


# %%
# ============================================================================
# 5. LINEAR REGRESSION WRAPPER FOR CLASSIFICATION
# ============================================================================
# Linear regression was not designed for classification. To produce a
# class prediction from a continuous output we encode the target as
# an ordinal {0, 1, 2}, fit ordinary least squares, and round the
# prediction to the closest valid class index. This is an intentionally
# weak baseline: it imposes an arbitrary order on a nominal target
# and cannot represent non-linear decision boundaries.
class LinearRegressionClassifier:
    """
    Turns sklearn's LinearRegression into a classifier by rounding
    continuous predictions to the nearest class index. Used ONLY as a
    pedagogical baseline to illustrate why linear regression is
    inappropriate for multiclass classification.
    """

    def __init__(self, class_order: list[str]):
        self.class_order = class_order
        self.cls2int = {c: i for i, c in enumerate(class_order)}
        self.int2cls = {i: c for c, i in self.cls2int.items()}
        self.model = LinearRegression()

    def fit(self, X, y_str):
        y_int = np.array([self.cls2int[c] for c in y_str], dtype=float)
        self.model.fit(X, y_int)
        return self

    def predict(self, X):
        y_cont = self.model.predict(X)
        # Round to nearest integer and clip to the valid class range.
        y_int = np.clip(np.round(y_cont), 0, len(self.class_order) - 1)
        return np.array([self.int2cls[int(i)] for i in y_int])


# %%
# ============================================================================
# 6. MODEL FACTORY
# ============================================================================
# Every model is rebuilt from scratch for each (model, label-version)
# combination so that state from one training run cannot leak into
# the next. class_weight='balanced' is used where supported to cope
# with the natural class imbalance of the dataset.
def build_models() -> dict:
    return {
        "DecisionTree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        "LinearRegression": LinearRegressionClassifier(CLASS_ORDER),
    }


# %%
# ============================================================================
# 7. EVALUATION HELPER
# ============================================================================
def evaluate(y_true_str, y_pred_str, model_name: str, label_source: str) -> dict:
    """
    Compute the suite of metrics reported for every (model, label)
    pair. We report macro-averaged metrics and also the Dropout-class
    recall and F1, because Dropout is the class of primary interest
    (the one we want to catch early).
    """
    acc = accuracy_score(y_true_str, y_pred_str)
    prec_macro = precision_score(
        y_true_str, y_pred_str, average="macro",
        labels=CLASS_ORDER, zero_division=0,
    )
    rec_macro = recall_score(
        y_true_str, y_pred_str, average="macro",
        labels=CLASS_ORDER, zero_division=0,
    )
    f1_macro = f1_score(
        y_true_str, y_pred_str, average="macro",
        labels=CLASS_ORDER, zero_division=0,
    )
    rec_dropout = recall_score(
        y_true_str, y_pred_str,
        labels=["Dropout"], average="macro", zero_division=0,
    )
    f1_dropout = f1_score(
        y_true_str, y_pred_str,
        labels=["Dropout"], average="macro", zero_division=0,
    )
    return {
        "model":        model_name,
        "label_source": label_source,
        "accuracy":     acc,
        "precision_macro": prec_macro,
        "recall_macro":    rec_macro,
        "f1_macro":        f1_macro,
        "recall_dropout":  rec_dropout,
        "f1_dropout":      f1_dropout,
    }


# %%
# ============================================================================
# 8. TRAIN AND EVALUATE EVERY (MODEL, LABEL SOURCE) COMBINATION
# ============================================================================
# Six training runs total: 3 models x 2 label sources. We store
# metrics, predictions and confusion matrices for every combination.
label_sources = {
    "original":  y_train_orig,
    "corrected": y_train_corr,
}

metrics_rows: list[dict] = []
confusion_mats: dict[tuple[str, str], np.ndarray] = {}
predictions: dict[tuple[str, str], np.ndarray] = {}

for label_src, y_train in label_sources.items():
    print(f"\n--- Training with {label_src} labels "
          f"({len(set(y_train))} classes) ---")
    models = build_models()
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        # Evaluation is ALWAYS against the original test labels.
        row = evaluate(y_test_orig, y_pred, name, label_src)
        metrics_rows.append(row)
        predictions[(name, label_src)] = y_pred
        confusion_mats[(name, label_src)] = confusion_matrix(
            y_test_orig, y_pred, labels=CLASS_ORDER,
        )
        print(f"  {name:<20} acc={row['accuracy']:.3f}  "
              f"f1_macro={row['f1_macro']:.3f}  "
              f"recall_dropout={row['recall_dropout']:.3f}")

metrics_df = pd.DataFrame(metrics_rows)
print("\n[All metrics]")
print(metrics_df.round(3).to_string(index=False))


# %%
# ============================================================================
# 9. CONFUSION MATRICES (3x2 grid)
# ============================================================================
fig, axes = plt.subplots(3, 2, figsize=(11, 13))
model_names = ["DecisionTree", "LogisticRegression", "LinearRegression"]
for i, name in enumerate(model_names):
    for j, label_src in enumerate(["original", "corrected"]):
        ax = axes[i, j]
        cm = confusion_mats[(name, label_src)]
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER,
            cbar=False, ax=ax,
        )
        metric_row = metrics_df[
            (metrics_df["model"] == name)
            & (metrics_df["label_source"] == label_src)
        ].iloc[0]
        ax.set_title(
            f"{name}  |  {label_src} labels\n"
            f"acc={metric_row['accuracy']:.3f}  "
            f"f1_macro={metric_row['f1_macro']:.3f}",
            fontsize=10,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
fig.suptitle(
    "Confusion matrices on the original test set\n"
    "(rows = true label, columns = predicted label)",
    fontsize=13, y=1.00,
)
save_figure(fig, "11_confusion_matrices_grid")


# %%
# ============================================================================
# 10. METRIC COMPARISON BAR CHART
# ============================================================================
# Side-by-side bars comparing each model under both label sources,
# focused on the metrics that matter for this problem: macro F1 and
# recall on the Dropout class.
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
metrics_to_plot = [
    ("f1_macro",       "Macro F1"),
    ("recall_dropout", "Recall on Dropout class"),
]
x_pos = np.arange(len(model_names))
bar_width = 0.35

for ax, (col, title) in zip(axes, metrics_to_plot):
    orig_vals = [
        metrics_df[(metrics_df["model"] == m)
                   & (metrics_df["label_source"] == "original")][col].iloc[0]
        for m in model_names
    ]
    corr_vals = [
        metrics_df[(metrics_df["model"] == m)
                   & (metrics_df["label_source"] == "corrected")][col].iloc[0]
        for m in model_names
    ]
    ax.bar(x_pos - bar_width / 2, orig_vals, bar_width,
           label="Original labels", color="#7f7f7f", edgecolor="black")
    ax.bar(x_pos + bar_width / 2, corr_vals, bar_width,
           label="Corrected labels", color="#1f77b4", edgecolor="black")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=15)
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.set_ylim(0, 1.0)
    ax.legend()
    # Print values on top of bars
    for i, (o, c) in enumerate(zip(orig_vals, corr_vals)):
        ax.text(i - bar_width / 2, o + 0.01, f"{o:.2f}",
                ha="center", fontsize=9)
        ax.text(i + bar_width / 2, c + 0.01, f"{c:.2f}",
                ha="center", fontsize=9)

fig.suptitle("Impact of label correction on test-set performance",
             fontsize=13, y=1.02)
save_figure(fig, "12_metric_comparison_barchart")


# %%
# ============================================================================
# 11. DECISION TREE FEATURE IMPORTANCES
# ============================================================================
# Decision trees expose interpretable feature importances. We report
# them for both label variants to check whether relabeling changed
# which features the tree relies on. Useful for the discussion in
# the slides: are the dominant predictors stable across the two
# training sets?
feature_names = list(X_all.columns)

fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True)
for ax, label_src in zip(axes, ["original", "corrected"]):
    dt = build_models()["DecisionTree"]
    dt.fit(X_train_s, label_sources[label_src])
    importances = pd.Series(dt.feature_importances_, index=feature_names)
    top = importances.sort_values(ascending=True).tail(10)
    ax.barh(top.index, top.values, color="#1f77b4", edgecolor="black")
    ax.set_title(f"Decision Tree feature importances\n"
                 f"({label_src} labels)")
    ax.set_xlabel("Importance")
fig.suptitle("Top 10 features by Decision Tree importance", y=1.02)
save_figure(fig, "13_decision_tree_feature_importances")


# %%
# ============================================================================
# 12. PERSIST METRICS FOR POINT 5
# ============================================================================
metrics_df.round(4).to_csv(
    OUTPUTS_DIR / "supervised_metrics.csv", index=False,
)

# Also persist the raw confusion matrices as a single npz for Point 5.
cm_dict = {
    f"{name}__{label_src}": confusion_mats[(name, label_src)]
    for name in model_names
    for label_src in ["original", "corrected"]
}
np.savez(OUTPUTS_DIR / "confusion_matrices.npz", **cm_dict)

print("\n[Artifacts saved]")
print(f"  {OUTPUTS_DIR / 'supervised_metrics.csv'}")
print(f"  {OUTPUTS_DIR / 'confusion_matrices.npz'}")


# %%
# ============================================================================
# 13. SUMMARY
# ============================================================================
best = metrics_df.sort_values("f1_macro", ascending=False).iloc[0]
print("\n" + "=" * 72)
print("SUPERVISED TRAINING SUMMARY")
print("=" * 72)
print(f"- Combinations trained : {len(metrics_df)}")
print(f"- Best model overall   : {best['model']} ({best['label_source']} labels)")
print(f"  - accuracy       : {best['accuracy']:.3f}")
print(f"  - macro F1       : {best['f1_macro']:.3f}")
print(f"  - recall_dropout : {best['recall_dropout']:.3f}")
print("=" * 72)