# Early Prediction of University Academic Dropout

## Project Overview

This academic project developed for the course **Fundamentals of Machine Learning (SI3015)**, focused on demonstrating the complete Machine Learning lifecycle through an educational use case.

The selected problem addresses the **early prediction of academic dropout in higher education** using supervised Machine Learning techniques. The motivation behind the project is based on real-world scenarios where:

- Dropout in higher education is a critical challenge with deep social and institutional impact.
- In Colombia, SPADIES reported a **23.15% dropout rate at the university level** for 2023.
- Dropout implies a loss of opportunities for the student, reduces the efficiency of the educational system, and demands the design of **early data-driven interventions** (alerts, tutoring, financial support).
- Static rule-based systems are insufficient to capture the complex, multifactorial nature of dropout risk.

The project emphasizes conceptual understanding, justification of technical decisions, and realistic system design rather than maximizing predictive performance.

**Student:** Alejandro Garcés Ramírez

---

## Problem Definition

The problem is defined as:

> Identifying students at high risk of academic dropout during the early stages of their university trajectory, enabling timely institutional intervention.

### Key Characteristics of the Problem

- **Supervised learning** — classification task (Dropout / Graduate / Enrolled).
- Historical labeled data is available with known student outcomes.
- Multiple heterogeneous data sources contribute to the prediction.
- The model is designed as a **decision-support tool**, not an autonomous decision-maker.

### Why Machine Learning?

- Traditional threshold-based rules (e.g., "GPA below 2.5") fail to capture complex interactions between academic, socioeconomic, and institutional variables.
- ML models can learn non-linear patterns from historical data.
- Supervised classification allows explicit evaluation against known outcomes.

### Success Criteria

- High **recall** for the dropout class — minimizing undetected at-risk students is a priority.
- Reasonable **precision** — avoiding excessive false alarms that would overwhelm institutional resources.
- **F1-score** as the balanced metric for model selection.
- Interpretability of predictions to support human decision-making.
- Practical integration into an early warning system workflow.

---

## ML Project Lifecycle

This project follows the standard Machine Learning lifecycle:

### 1. Problem Definition and Scoping

Understanding the institutional context, defining what constitutes "dropout," and establishing that the model will serve as an advisory tool for academic counselors — not as an automated decision system.

### 2. Data Collection and Quality Assessment

**Data Sources:**

- **Academic data:** Grades, attendance, credits enrolled and approved, course repetition history, academic trend across semesters.
- **Socioeconomic data:** Employment status, financial aid, family context, tuition status.
- **Institutional data:** Use of tutoring services, interactions with student welfare offices.

**Bias Risks Identified:**

- **Temporal bias:** Changes in academic policies or historical contexts that shift the meaning of variables over time.
- **Cognitive bias:** Inconsistent definitions of "dropout" across departments or time periods.
- **Spatial bias:** Differences between programs, faculties, or campuses that may skew model generalization.

**Diversity and Entropy:**

- Data diversity ensures the model captures predominant patterns rather than artifacts.
- Higher entropy in the training data represents the real variability of the phenomenon, avoiding fragile or overfitted models.

### 3. Data Preparation and Feature Engineering

**Data Cleaning:**

- Handling missing values without automatically eliminating at-risk students — imputation using robust measures (median).
- Outlier detection is performed for exploratory analysis only, not automatic removal.

**Feature Engineering — Derived Variables:**

- `approval_ratio` = credits approved / credits enrolled (1st and 2nd semester)
- `academic_trend` = GPA change between semesters
- `work_load_binary` = employed / not employed (derived from employment status)

**Normalization and Encoding:**

- Numerical variables are scaled to the [0, 1] range (MinMaxScaler).
- Categorical variables are encoded without imposing artificial order (One-Hot Encoding).

**Data Splitting:**

Data is strategically divided into three sets:
- **Training** (70%) — learning patterns.
- **Validation** (15%) — hyperparameter tuning.
- **Test** (15%) — final objective evaluation.

### 4. Model Training and Evaluation

**Candidate Models:**

| Model | Rationale |
|---|---|
| Logistic Regression | Interpretable linear baseline for binary/multiclass classification |
| Decision Tree | Captures non-linear relationships; intuitive visualization |
| Random Forest | Robust ensemble that reduces overfitting through bagging |

**Evaluation Metrics:**

- **Precision:** Proportion of correct dropout predictions among all predicted dropouts.
- **Recall:** Ability to identify all actual dropout cases.
- **F1-Score:** Harmonic mean balancing precision and recall.
- **Confusion Matrix:** Detailed view of classification errors per class.
- **Cross-validation:** K-Fold to assess generalization stability.

### 5. Deployment and Usage Perspective

**Pre-deployment Validation:**

The model is never deployed without exhaustive evaluation on test data and validation with institutional stakeholders.

**Usage as Decision Support:**

The system generates early alerts that academic advisors analyze together with additional contextual information. The ML model supports pattern-based human decisions — **final responsibility always remains with trained professionals.**

**Interpretation Workshop:**

A practical exercise with a fictitious student case: analysis of prediction, contributing factors, and discussion of possible interventions.

---

## Dataset

This implementation uses the **[Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)** dataset from the UCI Machine Learning Repository.

The dataset was created from higher education institutions and includes information known at the time of student enrollment — such as academic path, demographics, and socioeconomic factors — along with students' academic performance at the end of the first and second semesters. The task is to classify students into three categories: **Dropout**, **Enrolled**, or **Graduate**.

### Dataset Characteristics

- **4424 instances** (students)
- **36 features** including demographic, socioeconomic, macroeconomic, and academic variables
- **3 target classes:** Dropout, Enrolled, Graduate
- Mixed numerical and categorical variables

---

## Project Structure

```
early_prediction_university_academic_dropout/
├── README.md                  # This file — project documentation
└── src/
    └── dropout_prediction.py  # Full ML pipeline implementation
```

---

## Implementation Details

The `src/dropout_prediction.py` script implements the complete ML lifecycle:

1. **Data loading and exploration** — EDA, class distribution, descriptive statistics.
2. **Data cleaning** — Missing value handling, type validation.
3. **Feature engineering** — Derived variables, approval ratios, academic trends.
4. **Preprocessing** — Scaling (MinMaxScaler), encoding, stratified train/validation/test split.
5. **Model training** — Logistic Regression, Decision Tree, and Random Forest.
6. **Evaluation** — Classification reports, confusion matrices, cross-validation, feature importance.
7. **Interpretation example** — Simulated student case analysis with contributing factor breakdown.

---

## How to Run

```bash
# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn

# Run the pipeline
cd src/
python dropout_prediction.py
```

The script will generate all outputs to the console, including EDA summaries, model comparison results, and a simulated student risk assessment.

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

## Key Concepts Covered

- Supervised learning (multiclass classification)
- X, y data representation
- Data quality and bias assessment
- Feature engineering and derived variables
- Feature scaling and categorical encoding
- Model selection and comparison
- Evaluation metrics (Precision, Recall, F1-Score)
- Cross-validation for generalization
- Model interpretability and feature importance
- Human-in-the-loop decision support systems

---

## Core Principle

> The model learns patterns from historical data and does not make autonomous decisions. Its function is to assist the human decision-making process. Machine Learning supports pattern-based human decisions — final responsibility always remains with trained professionals.