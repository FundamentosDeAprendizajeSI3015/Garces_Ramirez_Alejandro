"""
Point 5 - Final comparison: original vs corrected labels.

This module is the closing step of the pipeline. It does not train any
model: it consumes the artifacts produced by Point 3 (relabel summary)
and Point 4 (supervised metrics) and synthesises a verdict that
answers the central research question of the project:

    "Does training on labels corrected by the unsupervised ensemble
     improve generalisation on the original test distribution?"

Outputs:
    - A side-by-side metrics table (original vs corrected) per model.
    - A delta table showing the absolute change in each metric.
    - A 'verdict' figure ready for the presentation.
    - A JSON summary aggregating everything for the speaker script.

Run:
    python src/05_compare.py
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

from utils import OUTPUTS_DIR, save_figure

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11

# Metrics ordered as they will appear in tables and figures.
METRIC_ORDER = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "recall_dropout",
    "f1_dropout",
]

print("=" * 72)
print("POINT 5 - FINAL COMPARISON: ORIGINAL vs CORRECTED LABELS")
print("=" * 72)


# %%
# ============================================================================
# 2. LOAD ARTIFACTS FROM PREVIOUS POINTS
# ============================================================================
metrics_path = OUTPUTS_DIR / "supervised_metrics.csv"
relabel_path = OUTPUTS_DIR / "relabel_summary.json"

for p in [metrics_path, relabel_path]:
    if not p.exists():
        raise FileNotFoundError(
            f"Missing {p}. Run Points 3 and 4 before this module."
        )

metrics_df = pd.read_csv(metrics_path)
with open(relabel_path) as f:
    relabel_summary = json.load(f)

print(f"\n[Loaded] supervised_metrics.csv  ({len(metrics_df)} rows)")
print(f"[Loaded] relabel_summary.json    "
      f"({relabel_summary['relabeled_total']} students relabeled)")


# %%
# ============================================================================
# 3. PIVOT METRICS INTO SIDE-BY-SIDE TABLE
# ============================================================================
# Long format -> wide format with one row per model and one pair of
# columns per metric (original / corrected). Easier to read and to
# embed in slides.
wide = metrics_df.pivot(
    index="model",
    columns="label_source",
    values=METRIC_ORDER,
)
# Reorder columns so each metric shows original next to corrected.
wide = wide.reindex(columns=pd.MultiIndex.from_product(
    [METRIC_ORDER, ["original", "corrected"]]
))
# Fixed model order for consistent reading.
model_order = ["DecisionTree", "LogisticRegression", "LinearRegression"]
wide = wide.reindex(model_order)

print("\n[Side-by-side metrics: original vs corrected]")
print(wide.round(3).to_string())


# %%
# ============================================================================
# 4. DELTA TABLE (CORRECTED - ORIGINAL)
# ============================================================================
# Positive delta = corrected labels improved the metric.
# Negative delta = corrected labels hurt the metric.
delta = pd.DataFrame(index=model_order, columns=METRIC_ORDER, dtype=float)
for model in model_order:
    for metric in METRIC_ORDER:
        orig = wide.loc[model, (metric, "original")]
        corr = wide.loc[model, (metric, "corrected")]
        delta.loc[model, metric] = corr - orig

print("\n[Delta table (corrected - original)]")
print(delta.round(3).to_string())

# Aggregated counts: how many (model, metric) pairs improved vs degraded.
improved = int((delta > 0).sum().sum())
degraded = int((delta < 0).sum().sum())
unchanged = int((delta == 0).sum().sum())
total = improved + degraded + unchanged
print(f"\n[Verdict] improved={improved}/{total}  "
      f"degraded={degraded}/{total}  unchanged={unchanged}/{total}")


# %%
# ============================================================================
# 5. VERDICT FIGURE - DELTA HEATMAP
# ============================================================================
# A diverging colormap centred at zero makes the verdict visually
# unmistakable: red cells = the corrected pipeline made things worse,
# green cells = the corrected pipeline helped, white = no change.
fig, ax = plt.subplots(figsize=(10, 4.5))
sns.heatmap(
    delta.astype(float),
    annot=True,
    fmt="+.3f",
    cmap="RdYlGn",
    center=0,
    vmin=-0.2,
    vmax=0.2,
    cbar_kws={"label": "Δ (corrected − original)"},
    linewidths=0.5,
    linecolor="white",
    ax=ax,
)
ax.set_title(
    "Verdict: impact of label correction on test-set metrics\n"
    "(green = corrected helped · red = corrected hurt)",
    fontsize=12,
)
ax.set_xlabel("")
ax.set_ylabel("")
save_figure(fig, "14_verdict_delta_heatmap")


# %%
# ============================================================================
# 6. WINNERS-PER-METRIC TABLE
# ============================================================================
# For every metric, identify which (model, label-source) combination
# achieved the highest value. This is the "leaderboard" view.
winners = []
for metric in METRIC_ORDER:
    row = metrics_df.loc[metrics_df[metric].idxmax()]
    winners.append({
        "metric":  metric,
        "winner":  f"{row['model']} ({row['label_source']})",
        "value":   round(row[metric], 4),
    })
winners_df = pd.DataFrame(winners)
print("\n[Winner per metric]")
print(winners_df.to_string(index=False))

# Count winners by label source: did "original" or "corrected" win
# more metrics overall?
winner_label = winners_df["winner"].str.extract(r"\((.*)\)")[0]
label_wins = winner_label.value_counts().to_dict()
print(f"\n[Wins by label source] {label_wins}")


# %%
# ============================================================================
# 7. RANKED MODELS BY MACRO F1
# ============================================================================
ranking = (
    metrics_df.sort_values("f1_macro", ascending=False)
    .reset_index(drop=True)
    [["model", "label_source", "accuracy", "f1_macro", "recall_dropout"]]
)
ranking.index += 1
print("\n[Global ranking by macro F1]")
print(ranking.round(3).to_string())


# %%
# ============================================================================
# 8. SAVE FINAL SUMMARY (SPEAKER SCRIPT INPUT)
# ============================================================================
final_summary = {
    "relabel": {
        "students_relabeled":  relabel_summary["relabeled_total"],
        "relabel_fraction":    relabel_summary["relabeled_fraction"],
        "signal_counts":       relabel_summary["signal_counts"],
    },
    # Flatten the pivoted MultiIndex columns into "metric__label_source"
    # so the dict is JSON-serialisable.
    "metrics_wide": {
        f"{metric}__{label}": wide[(metric, label)].round(4).to_dict()
        for metric in METRIC_ORDER
        for label in ["original", "corrected"]
    },
    "delta_corrected_minus_original": delta.round(4).to_dict(),
    "wins_by_label_source": label_wins,
    "verdict_counts": {
        "metrics_improved":  improved,
        "metrics_degraded":  degraded,
        "metrics_unchanged": unchanged,
        "metrics_total":     total,
    },
    "best_overall": {
        "model":           ranking.iloc[0]["model"],
        "label_source":    ranking.iloc[0]["label_source"],
        "accuracy":        round(float(ranking.iloc[0]["accuracy"]), 4),
        "f1_macro":        round(float(ranking.iloc[0]["f1_macro"]), 4),
        "recall_dropout":  round(float(ranking.iloc[0]["recall_dropout"]), 4),
    },
}
with open(OUTPUTS_DIR / "final_comparison.json", "w") as f:
    json.dump(final_summary, f, indent=2, default=str)

# Also persist the wide and delta tables as CSV for the slides.
wide.round(4).to_csv(OUTPUTS_DIR / "comparison_wide.csv")
delta.round(4).to_csv(OUTPUTS_DIR / "comparison_delta.csv")

print("\n[Artifacts saved]")
print(f"  {OUTPUTS_DIR / 'comparison_wide.csv'}")
print(f"  {OUTPUTS_DIR / 'comparison_delta.csv'}")
print(f"  {OUTPUTS_DIR / 'final_comparison.json'}")


# %%
# ============================================================================
# 9. NARRATIVE VERDICT
# ============================================================================
# Print a plain-text narrative that you can paste straight into the
# presentation script. The wording is conditional on the actual sign
# of the deltas, so the message is honest no matter how the numbers
# turn out on a re-run.
print("\n" + "=" * 72)
print("FINAL VERDICT")
print("=" * 72)

best = ranking.iloc[0]
print(f"\n- Best model overall      : {best['model']} "
      f"(trained on {best['label_source']} labels)")
print(f"- Best F1 macro           : {best['f1_macro']:.3f}")
print(f"- Best recall on Dropout  : {best['recall_dropout']:.3f}")

print(f"\n- Students relabeled by Point 3: "
      f"{relabel_summary['relabeled_total']} "
      f"({relabel_summary['relabeled_fraction']:.1%} of the dataset)")

if degraded > improved:
    print(
        "\n- Direction of effect : the corrected labels DEGRADED performance "
        f"in {degraded}/{total} metric-model pairs."
        "\n  Interpretation     : the students flagged as 'suspicious' by the"
        "\n                       unsupervised ensemble are not mislabeled --"
        "\n                       they are legitimate borderline cases. The"
        "\n                       experiment empirically validates the project"
        "\n                       principle: ML supports human decisions, it"
        "\n                       does not replace them."
    )
elif improved > degraded:
    print(
        "\n- Direction of effect : the corrected labels IMPROVED performance "
        f"in {improved}/{total} metric-model pairs."
        "\n  Interpretation     : the unsupervised ensemble successfully"
        "\n                       identified mislabeled training examples."
    )
else:
    print(
        "\n- Direction of effect : neutral. Improvements and degradations "
        "balance out."
    )

print("=" * 72)