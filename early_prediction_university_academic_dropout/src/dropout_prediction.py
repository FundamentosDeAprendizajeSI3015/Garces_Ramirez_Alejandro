"""
Early Prediction of University Academic Dropout
=================================================
Complete ML Pipeline Implementation

Course: Fundamentals of Machine Learning (SI3015)
Student: Alejandro Garcés Ramírez

This script implements the full Machine Learning lifecycle for predicting
student dropout in higher education, following supervised learning principles.

Dataset: Predict Students' Dropout and Academic Success (UCI ML Repository)
    - If the dataset CSV is available locally, it will be loaded automatically.
    - Otherwise, a realistic synthetic dataset is generated for demonstration.

Pipeline stages:
    1. Data Loading & Exploration (EDA)
    2. Data Cleaning & Quality Assessment
    3. Feature Engineering
    4. Preprocessing (Scaling, Encoding, Splitting)
    5. Model Training (Logistic Regression, Decision Tree, Random Forest)
    6. Evaluation (Classification Report, Confusion Matrix, Cross-Validation)
    7. Feature Importance & Interpretation
    8. Simulated Student Case Study
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

warnings.filterwarnings("ignore")
np.random.seed(42)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# 1. DATA LOADING
# ============================================================================

def load_or_generate_data():
    """
    Attempt to load the UCI dataset from a local CSV file.
    If unavailable, generate a realistic synthetic dataset that mirrors
    the structure described in the project (academic, socioeconomic,
    and institutional features).
    """
    csv_path = os.path.join(os.path.dirname(__file__), "data.csv")
    if os.path.exists(csv_path):
        print("[INFO] Loading dataset from local CSV...")
        df = pd.read_csv(csv_path, delimiter=";")
        return df

    print("[INFO] Local CSV not found. Generating synthetic dataset...")
    print("       (Place 'data.csv' in src/ to use the real UCI dataset)\n")

    n_samples = 4424
    rng = np.random.default_rng(42)

    # --- Academic features (1st and 2nd semester) ---
    enrolled_1st = rng.integers(4, 9, size=n_samples)
    approved_1st = np.clip(
        enrolled_1st - rng.poisson(1.2, size=n_samples), 0, enrolled_1st
    )
    grade_1st = np.clip(rng.normal(11.5, 3.0, size=n_samples), 0, 20)

    enrolled_2nd = rng.integers(4, 9, size=n_samples)
    approved_2nd = np.clip(
        enrolled_2nd - rng.poisson(1.0, size=n_samples), 0, enrolled_2nd
    )
    grade_2nd = np.clip(rng.normal(12.0, 2.8, size=n_samples), 0, 20)

    evaluations_1st = rng.integers(3, 12, size=n_samples)
    evaluations_2nd = rng.integers(3, 12, size=n_samples)

    # --- Demographic features ---
    age_at_enrollment = rng.integers(17, 55, size=n_samples)
    gender = rng.choice([0, 1], size=n_samples, p=[0.35, 0.65])
    marital_status = rng.choice([1, 2, 3, 4, 5], size=n_samples,
                                p=[0.65, 0.20, 0.05, 0.05, 0.05])
    displaced = rng.choice([0, 1], size=n_samples, p=[0.90, 0.10])
    international = rng.choice([0, 1], size=n_samples, p=[0.97, 0.03])

    # --- Socioeconomic features ---
    scholarship_holder = rng.choice([0, 1], size=n_samples, p=[0.75, 0.25])
    debtor = rng.choice([0, 1], size=n_samples, p=[0.88, 0.12])
    tuition_up_to_date = rng.choice([0, 1], size=n_samples, p=[0.12, 0.88])
    mothers_qualification = rng.integers(1, 35, size=n_samples)
    fathers_qualification = rng.integers(1, 35, size=n_samples)
    mothers_occupation = rng.integers(0, 20, size=n_samples)
    fathers_occupation = rng.integers(0, 20, size=n_samples)

    # --- Macroeconomic features ---
    unemployment_rate = np.clip(rng.normal(11.0, 2.5, size=n_samples), 5, 20)
    inflation_rate = np.clip(rng.normal(1.5, 1.0, size=n_samples), -1, 5)
    gdp = np.clip(rng.normal(1.0, 2.0, size=n_samples), -5, 5)

    # --- Institutional features ---
    application_mode = rng.integers(1, 18, size=n_samples)
    application_order = rng.integers(0, 9, size=n_samples)
    course = rng.choice(
        [33, 171, 8014, 9003, 9070, 9085, 9119, 9130, 9147, 9238,
         9254, 9500, 9556, 9670, 9773, 9853, 9991],
        size=n_samples,
    )
    daytime_evening = rng.choice([0, 1], size=n_samples, p=[0.67, 0.33])
    previous_qualification_grade = np.clip(
        rng.normal(130, 15, size=n_samples), 90, 190
    )
    admission_grade = np.clip(rng.normal(127, 13, size=n_samples), 90, 190)

    # --- Target variable ---
    # Create realistic target based on feature correlations
    risk_score = (
        -0.3 * (approved_1st / np.maximum(enrolled_1st, 1))
        - 0.3 * (approved_2nd / np.maximum(enrolled_2nd, 1))
        - 0.15 * (grade_1st / 20)
        - 0.15 * (grade_2nd / 20)
        + 0.1 * debtor
        - 0.1 * scholarship_holder
        + 0.05 * (age_at_enrollment > 25).astype(float)
        - 0.05 * tuition_up_to_date
        + rng.normal(0, 0.15, size=n_samples)
    )

    target = np.where(
        risk_score > np.percentile(risk_score, 68), "Dropout",
        np.where(risk_score < np.percentile(risk_score, 30), "Graduate", "Enrolled")
    )

    df = pd.DataFrame({
        "Marital status": marital_status,
        "Application mode": application_mode,
        "Application order": application_order,
        "Course": course,
        "Daytime/evening attendance": daytime_evening,
        "Previous qualification (grade)": previous_qualification_grade,
        "Mother's qualification": mothers_qualification,
        "Father's qualification": fathers_qualification,
        "Mother's occupation": mothers_occupation,
        "Father's occupation": fathers_occupation,
        "Displaced": displaced,
        "Debtor": debtor,
        "Tuition fees up to date": tuition_up_to_date,
        "Gender": gender,
        "Scholarship holder": scholarship_holder,
        "Age at enrollment": age_at_enrollment,
        "International": international,
        "Curricular units 1st sem (enrolled)": enrolled_1st,
        "Curricular units 1st sem (evaluations)": evaluations_1st,
        "Curricular units 1st sem (approved)": approved_1st,
        "Curricular units 1st sem (grade)": grade_1st,
        "Curricular units 2nd sem (enrolled)": enrolled_2nd,
        "Curricular units 2nd sem (evaluations)": evaluations_2nd,
        "Curricular units 2nd sem (approved)": approved_2nd,
        "Curricular units 2nd sem (grade)": grade_2nd,
        "Unemployment rate": unemployment_rate,
        "Inflation rate": inflation_rate,
        "GDP": gdp,
        "Admission grade": admission_grade,
        "Target": target,
    })

    return df


# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

def exploratory_analysis(df):
    """Perform and display exploratory data analysis."""
    print("=" * 70)
    print("  STAGE 1: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)

    print(f"\nDataset shape: {df.shape[0]} students x {df.shape[1]} features")
    print(f"\nData types:\n{df.dtypes.value_counts()}")
    print(f"\nMissing values per column:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    if df.isnull().sum().sum() == 0:
        print("  No missing values detected.")

    print(f"\n--- Target Class Distribution ---")
    target_counts = df["Target"].value_counts()
    for cls, count in target_counts.items():
        pct = count / len(df) * 100
        print(f"  {cls:12s}: {count:5d}  ({pct:.1f}%)")

    print(f"\n--- Descriptive Statistics (selected numerical features) ---")
    num_cols = [
        "Age at enrollment",
        "Admission grade",
        "Curricular units 1st sem (grade)",
        "Curricular units 2nd sem (grade)",
        "Unemployment rate",
    ]
    existing = [c for c in num_cols if c in df.columns]
    print(df[existing].describe().round(2).to_string())

    # -- Plots --
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Class distribution
    target_counts.plot(kind="bar", ax=axes[0], color=["#e74c3c", "#f39c12", "#27ae60"])
    axes[0].set_title("Target Class Distribution")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=0)

    # Grade distribution by target
    for label, color in zip(
        ["Dropout", "Enrolled", "Graduate"], ["#e74c3c", "#f39c12", "#27ae60"]
    ):
        subset = df[df["Target"] == label]
        if "Curricular units 1st sem (grade)" in subset.columns:
            axes[1].hist(
                subset["Curricular units 1st sem (grade)"],
                bins=30, alpha=0.5, label=label, color=color,
            )
    axes[1].set_title("1st Semester Grade Distribution by Target")
    axes[1].set_xlabel("Grade")
    axes[1].legend()

    # Age distribution
    if "Age at enrollment" in df.columns:
        df.boxplot(column="Age at enrollment", by="Target", ax=axes[2])
        axes[2].set_title("Age at Enrollment by Target")
        plt.suptitle("")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_eda_overview.png"), dpi=150)
    plt.close()
    print(f"\n  [SAVED] 01_eda_overview.png")

    return df


# ============================================================================
# 3. DATA CLEANING & QUALITY ASSESSMENT
# ============================================================================

def clean_data(df):
    """
    Clean the dataset following the project principles:
    - Handle missing values with median imputation (robust measure).
    - Detect outliers for analysis only — no automatic removal.
    - Validate data types.
    """
    print("\n" + "=" * 70)
    print("  STAGE 2: DATA CLEANING & QUALITY ASSESSMENT")
    print("=" * 70)

    df = df.copy()

    # Separate target
    target_col = "Target"

    # Impute missing numerical values with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    missing_before = df[num_cols].isnull().sum().sum()
    for col in num_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  Imputed '{col}' missing values with median = {median_val:.2f}")

    if missing_before == 0:
        print("  No missing numerical values to impute.")

    # Outlier detection (IQR method) — for reporting only
    print(f"\n--- Outlier Detection (IQR method — informational only) ---")
    outlier_report = {}
    for col in num_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        if n_outliers > 0:
            outlier_report[col] = n_outliers
    if outlier_report:
        for col, n in sorted(outlier_report.items(), key=lambda x: -x[1])[:10]:
            print(f"  {col:50s}: {n:4d} outliers detected")
        print("  NOTE: Outliers are retained to preserve at-risk student data.")
    else:
        print("  No significant outliers detected.")

    print(f"\n  Dataset after cleaning: {df.shape}")
    return df


# ============================================================================
# 4. FEATURE ENGINEERING
# ============================================================================

def engineer_features(df):
    """
    Create derived variables as described in the project:
    - Approval ratio (credits approved / credits enrolled)
    - Academic trend (grade change between semesters)
    - Work-related binary indicators
    """
    print("\n" + "=" * 70)
    print("  STAGE 3: FEATURE ENGINEERING")
    print("=" * 70)

    df = df.copy()

    # Approval ratio — 1st semester
    if all(c in df.columns for c in [
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (enrolled)",
    ]):
        df["approval_ratio_1st"] = (
            df["Curricular units 1st sem (approved)"]
            / df["Curricular units 1st sem (enrolled)"].replace(0, np.nan)
        ).fillna(0)
        print("  Created: approval_ratio_1st = approved / enrolled (1st sem)")

    # Approval ratio — 2nd semester
    if all(c in df.columns for c in [
        "Curricular units 2nd sem (approved)",
        "Curricular units 2nd sem (enrolled)",
    ]):
        df["approval_ratio_2nd"] = (
            df["Curricular units 2nd sem (approved)"]
            / df["Curricular units 2nd sem (enrolled)"].replace(0, np.nan)
        ).fillna(0)
        print("  Created: approval_ratio_2nd = approved / enrolled (2nd sem)")

    # Academic trend — grade change between semesters
    if all(c in df.columns for c in [
        "Curricular units 1st sem (grade)",
        "Curricular units 2nd sem (grade)",
    ]):
        df["academic_trend"] = (
            df["Curricular units 2nd sem (grade)"]
            - df["Curricular units 1st sem (grade)"]
        )
        print("  Created: academic_trend = grade_2nd - grade_1st")

    # Combined approval ratio
    if "approval_ratio_1st" in df.columns and "approval_ratio_2nd" in df.columns:
        df["avg_approval_ratio"] = (
            df["approval_ratio_1st"] + df["approval_ratio_2nd"]
        ) / 2
        print("  Created: avg_approval_ratio = mean of both semester ratios")

    # Debtor status is already binary; useful as-is
    if "Debtor" in df.columns:
        print("  Retained: Debtor (binary — equivalent to financial risk flag)")

    print(f"\n  Total features after engineering: {df.shape[1]}")
    return df


# ============================================================================
# 5. PREPROCESSING (Scaling, Encoding, Splitting)
# ============================================================================

def preprocess(df):
    """
    Prepare data for modeling:
    - Encode target variable.
    - Scale numerical features with MinMaxScaler to [0, 1].
    - Stratified split into train (70%), validation (15%), test (15%).
    """
    print("\n" + "=" * 70)
    print("  STAGE 4: PREPROCESSING")
    print("=" * 70)

    df = df.copy()

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(df["Target"])
    class_names = le.classes_
    print(f"  Target classes encoded: {dict(zip(class_names, le.transform(class_names)))}")

    X = df.drop(columns=["Target"])

    # Identify and encode any remaining categorical columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        print(f"  One-Hot encoding categorical columns: {cat_cols}")
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Scale numerical features to [0, 1]
    scaler = MinMaxScaler()
    feature_names = X.columns.tolist()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=feature_names)
    print(f"  Scaled {len(feature_names)} features to [0, 1] range (MinMaxScaler)")

    # Stratified split: 70% train, 15% validation, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"\n  Data split (stratified):")
    print(f"    Training:   {X_train.shape[0]:5d} samples ({X_train.shape[0]/len(X)*100:.0f}%)")
    print(f"    Validation: {X_val.shape[0]:5d} samples ({X_val.shape[0]/len(X)*100:.0f}%)")
    print(f"    Test:       {X_test.shape[0]:5d} samples ({X_test.shape[0]/len(X)*100:.0f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test, class_names, feature_names


# ============================================================================
# 6. MODEL TRAINING & EVALUATION
# ============================================================================

def train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test,
                       class_names, feature_names):
    """
    Train three candidate models and evaluate them:
    - Logistic Regression (interpretable linear baseline)
    - Decision Tree (non-linear, visual)
    - Random Forest (robust ensemble)
    """
    print("\n" + "=" * 70)
    print("  STAGE 5: MODEL TRAINING & EVALUATION")
    print("=" * 70)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, random_state=42, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=42,
            class_weight="balanced", n_jobs=-1,
        ),
    }

    results = {}

    for name, model in models.items():
        print(f"\n--- {name} ---")

        # Train
        model.fit(X_train, y_train)

        # Validate
        y_val_pred = model.predict(X_val)
        val_f1 = f1_score(y_val, y_val_pred, average="weighted")
        print(f"  Validation F1 (weighted): {val_f1:.4f}")

        # Test
        y_test_pred = model.predict(X_test)
        test_f1 = f1_score(y_test, y_test_pred, average="weighted")
        print(f"  Test F1 (weighted):       {test_f1:.4f}")

        print(f"\n  Classification Report (Test Set):")
        report = classification_report(y_test, y_test_pred,
                                       target_names=class_names, digits=3)
        print(report)

        # Cross-validation on training data
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                    scoring="f1_weighted")
        print(f"  5-Fold CV F1 (weighted): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        results[name] = {
            "model": model,
            "val_f1": val_f1,
            "test_f1": test_f1,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "y_test_pred": y_test_pred,
        }

    # -- Confusion matrices --
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, res["y_test_pred"])
        disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title(f"{name}\nTest F1={res['test_f1']:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_confusion_matrices.png"), dpi=150)
    plt.close()
    print(f"\n  [SAVED] 02_confusion_matrices.png")

    # -- Model comparison bar chart --
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(results.keys())
    val_scores = [results[n]["val_f1"] for n in names]
    test_scores = [results[n]["test_f1"] for n in names]
    x = np.arange(len(names))
    width = 0.3
    ax.bar(x - width / 2, val_scores, width, label="Validation F1", color="#3498db")
    ax.bar(x + width / 2, test_scores, width, label="Test F1", color="#2ecc71")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Weighted F1 Score")
    ax.set_title("Model Comparison — Weighted F1 Score")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_model_comparison.png"), dpi=150)
    plt.close()
    print(f"  [SAVED] 03_model_comparison.png")

    # Select best model
    best_name = max(results, key=lambda n: results[n]["test_f1"])
    print(f"\n  *** Best model: {best_name} (Test F1 = {results[best_name]['test_f1']:.4f}) ***")

    return results, best_name


# ============================================================================
# 7. FEATURE IMPORTANCE & INTERPRETATION
# ============================================================================

def feature_importance_analysis(results, best_name, feature_names):
    """Analyze and plot feature importances from the best model."""
    print("\n" + "=" * 70)
    print("  STAGE 6: FEATURE IMPORTANCE & INTERPRETATION")
    print("=" * 70)

    model = results[best_name]["model"]

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).mean(axis=0)
    else:
        print("  Model does not expose feature importances.")
        return

    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    top_n = min(15, len(feat_imp))

    print(f"\n  Top {top_n} features ({best_name}):")
    for i, (feat, imp) in enumerate(feat_imp.head(top_n).items(), 1):
        print(f"    {i:2d}. {feat:50s}  {imp:.4f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    feat_imp.head(top_n).plot(kind="barh", ax=ax, color="#2c3e50")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances — {best_name}")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_feature_importance.png"), dpi=150)
    plt.close()
    print(f"\n  [SAVED] 04_feature_importance.png")


# ============================================================================
# 8. SIMULATED STUDENT CASE STUDY (Interpretation Workshop)
# ============================================================================

def simulated_case_study(results, best_name, feature_names, class_names):
    """
    Simulate a fictitious student case to demonstrate how the model
    would be used as a decision-support tool by academic advisors.
    """
    print("\n" + "=" * 70)
    print("  STAGE 7: SIMULATED STUDENT CASE STUDY")
    print("=" * 70)

    model = results[best_name]["model"]

    # Create a fictitious at-risk student profile
    print("\n  --- Fictitious Student Profile ---")
    print("  Name:                  Ana María (fictitious)")
    print("  Age at enrollment:     22")
    print("  Scholarship holder:    No")
    print("  Debtor:                Yes")
    print("  Tuition up to date:    No")
    print("  1st sem enrolled:      6 units")
    print("  1st sem approved:      3 units")
    print("  1st sem grade avg:     8.5 / 20")
    print("  2nd sem enrolled:      5 units")
    print("  2nd sem approved:      2 units")
    print("  2nd sem grade avg:     7.0 / 20")

    # Build a feature vector matching the training features
    student_data = pd.DataFrame(np.zeros((1, len(feature_names))),
                                columns=feature_names)

    # Fill known values (normalized to [0, 1] range approximately)
    feature_map = {
        "Age at enrollment": 22 / 55,
        "Scholarship holder": 0.0,
        "Debtor": 1.0,
        "Tuition fees up to date": 0.0,
        "Gender": 1.0,
        "Displaced": 0.0,
        "International": 0.0,
        "Curricular units 1st sem (enrolled)": 6 / 9,
        "Curricular units 1st sem (approved)": 3 / 9,
        "Curricular units 1st sem (grade)": 8.5 / 20,
        "Curricular units 2nd sem (enrolled)": 5 / 9,
        "Curricular units 2nd sem (approved)": 2 / 9,
        "Curricular units 2nd sem (grade)": 7.0 / 20,
        "Admission grade": 110 / 190,
        "Previous qualification (grade)": 110 / 190,
        "Unemployment rate": 12 / 20,
        "Inflation rate": 1.5 / 5,
        "GDP": 0.5,
    }

    # Derived features
    feature_map["approval_ratio_1st"] = 3 / 6
    feature_map["approval_ratio_2nd"] = 2 / 5
    feature_map["academic_trend"] = (7.0 - 8.5) / 20
    feature_map["avg_approval_ratio"] = (3/6 + 2/5) / 2

    for feat, val in feature_map.items():
        if feat in student_data.columns:
            student_data[feat] = val

    # Predict
    prediction = model.predict(student_data)[0]
    predicted_class = class_names[prediction]

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(student_data)[0]
        print(f"\n  --- Model Prediction ---")
        for cls, prob in zip(class_names, probas):
            bar = "█" * int(prob * 30)
            print(f"    {cls:12s}: {prob*100:5.1f}%  {bar}")
    else:
        print(f"\n  --- Model Prediction ---")

    print(f"\n  Predicted outcome: ** {predicted_class} **")

    print(f"\n  --- Recommended Actions for Academic Advisor ---")
    if predicted_class == "Dropout":
        print("  ⚠  HIGH RISK — Immediate intervention recommended:")
        print("     • Schedule individual tutoring sessions")
        print("     • Evaluate financial aid eligibility")
        print("     • Connect with student welfare services")
        print("     • Reduce academic load next semester")
        print("     • Assign a faculty mentor")
    elif predicted_class == "Enrolled":
        print("  ⚡ MODERATE RISK — Monitoring recommended:")
        print("     • Follow up on academic progress mid-semester")
        print("     • Offer optional tutoring resources")
        print("     • Review financial situation periodically")
    else:
        print("  ✅ LOW RISK — Standard monitoring:")
        print("     • Continue regular academic tracking")

    print("\n  NOTE: This prediction is a decision-support tool.")
    print("  Final decisions must always be made by trained professionals")
    print("  considering the full student context.")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("╔" + "═" * 68 + "╗")
    print("║  EARLY PREDICTION OF UNIVERSITY ACADEMIC DROPOUT               ║")
    print("║  Complete ML Pipeline — Fundamentals of Machine Learning       ║")
    print("║  Student: Alejandro Garcés Ramírez                             ║")
    print("╚" + "═" * 68 + "╝\n")

    # 1. Load data
    df = load_or_generate_data()

    # 2. EDA
    df = exploratory_analysis(df)

    # 3. Clean
    df = clean_data(df)

    # 4. Feature engineering
    df = engineer_features(df)

    # 5. Preprocess
    X_train, X_val, X_test, y_train, y_val, y_test, class_names, feature_names = (
        preprocess(df)
    )

    # 6. Train & evaluate
    results, best_name = train_and_evaluate(
        X_train, X_val, X_test, y_train, y_val, y_test, class_names, feature_names
    )

    # 7. Feature importance
    feature_importance_analysis(results, best_name, feature_names)

    # 8. Simulated case study
    simulated_case_study(results, best_name, feature_names, class_names)

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print(f"  Output charts saved to: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 70)


if __name__ == "__main__":
    main()