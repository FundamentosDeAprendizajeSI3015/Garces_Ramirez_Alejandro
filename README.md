# **Fundamentals of Machine Learning – SI3015**

**Student: Alejandro Garcés Ramírez**

This repository contains the activities developed during the course Fundamentals of Machine Learning, organized by weeks and practical projects.

The main focus of the course was data exploration, analysis, and preprocessing, understanding that the quality of data representation (X, y) is a critical step before training Machine Learning models.

---

## **Repository Objective**

The goal of this repository is to document the complete data preparation process for Machine Learning, including:

- Initial dataset exploration (EDA)
- Data quality evaluation
- Descriptive statistics
- Outlier detection
- Pattern visualization
- Variable transformation and encoding
- Feature scaling
- Construction of datasets ready for modeling

The repository reflects the actual workflow followed during the course, prioritizing the understanding of the data pipeline before model development.

---

## **Week 2 — Iris Dataset Analysis**

**Dataset:** Iris Dataset (scikit-learn)

### **Objective**

Introduce the basic Machine Learning workflow, from data collection to initial model evaluation in a supervised learning setting.

### **Activities Performed**

- Dataset loading and DataFrame construction
- Initial data exploration (EDA)
- Class distribution analysis
- Descriptive statistics
- Feature visualization and correlation analysis
- Train/Test split
- Feature scaling using StandardScaler
- Model training and evaluation

### **Concepts Covered**

- X, y representation
- Feature scaling
- Exploratory data analysis
- Supervised classification fundamentals

---

## **Week 3 — Fintech Dataset: Data Preprocessing**

**Dataset:** Fintech dataset

### **Objective**

Build a data exploration and preprocessing pipeline, focusing on understanding that model performance strongly depends on the quality of the prepared dataset.

### **Activities Performed**

- Dataset loading and exploration
- Data cleaning and transformation
- Feature analysis and visualization
- Preprocessing pipeline construction

### **Concepts Covered**

- Data quality assessment
- Data cleaning workflows
- Feature transformation
- Separation between preprocessing and modeling

---

## **Week 4 — Movies Dataset: Data Exploration and Transformation**

**Dataset:** movies.csv

### **Objective**

Develop a complete data cleaning and transformation workflow using a dataset with real-world data quality issues.

### **Activities Performed**

- Data cleaning and normalization
- Column name normalization
- Data type conversion
- Handling missing values
- Descriptive statistics (mean, median, mode, standard deviation, variance, IQR)
- Outlier detection
- Feature transformation (Label Encoding, One-Hot Encoding)
- Correlation analysis
- Feature scaling (MinMaxScaler, StandardScaler)
- Output generation of cleaned dataset

### **Concepts Covered**

- Data quality challenges in real datasets
- Basic feature engineering
- Statistical noise reduction
- Preparation of tabular data for supervised learning

---

## **Project — Early Prediction of University Academic Dropout**

### **Project Overview**

Academic project focused on demonstrating the complete Machine Learning lifecycle through an educational use case. The selected problem addresses the early prediction of academic dropout in higher education using supervised learning techniques.

In Colombia, SPADIES reported a 23.15% dropout rate at the university level for 2023. The project aims to build a model that identifies students at risk of dropping out, enabling timely institutional interventions such as tutoring, financial aid, and academic advising.

### **Key Aspects**

- Supervised classification (Dropout / Enrolled / Graduate)
- Feature engineering: approval ratios, academic trends, socioeconomic indicators
- Candidate models: Logistic Regression, Decision Tree, Random Forest
- Evaluation: Precision, Recall, F1-Score, Confusion Matrix, Cross-Validation
- Simulated student case study for interpretation
- Human-in-the-loop decision support design

### **Core Principle**

> The model learns patterns from historical data and does not make autonomous decisions. Its function is to assist the human decision-making process. Final responsibility always remains with trained professionals.

Full documentation available in [`early_prediction_university_academic_dropout/README.md`](early_prediction_university_academic_dropout/README.md).

---

## **Technologies Used**

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

## **General Course Conclusions**

- Model performance heavily depends on data preprocessing quality.
- Visual exploration helps understand the geometric structure of datasets.
- Scaling and feature transformation are essential steps in ML workflows.
- Separating preprocessing from modeling improves reproducibility.
- Real-world datasets require significantly more cleaning than academic datasets.
- Machine Learning models should support human decisions, not replace them.