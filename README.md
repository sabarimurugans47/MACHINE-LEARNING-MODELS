# 🧠 Stroke Prediction using Machine Learning

This project focuses on predicting the likelihood of a stroke in individuals based on clinical and demographic features using various classification algorithms.

---

## 📌 Project Overview

The goal of this project is to build a predictive machine learning model that can identify individuals at risk of stroke using features such as age, glucose level, heart disease, smoking status, hypertension, and more. This can help in early detection and preventive care in healthcare systems.

---

## 📂 Dataset

- 📁 Source: [Kaggle - Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- 🎯 Features:
  - age, hypertension, heart_disease, avg_glucose_level, smoking_status, etc.
- 🔻 Target:
  - stroke (1 - Stroke occurred, 0 - No stroke)

---

## 🧪 ML Workflow

1. *Data Preprocessing*
   - Handling missing values
   - Label encoding (for categorical features)
   - Feature scaling
   - SMOTE for class imbalance

2. *Feature Selection*
   - Correlation analysis
   - Lasso regression for shrinkage

3. *Model Building*
   - Logistic Regression
   - Random Forest Classifier
   - XGBoost Classifier

4. *Model Evaluation*
   - Accuracy, Precision, Recall, F1-score
   - ROC-AUC Curve
   - Threshold tuning

5. *Prediction*
   - Tested on new synthetic input data
   - Output: Predicted risk (High / Low Stroke Risk)

---

## 📊 Results

- ✅ Best Model: *XGBoost Classifier*
- 🎯 Accuracy: *92%*
- 🔎 Best Threshold: *0.61* (custom-tuned)

---

## 📌 Key Libraries Used

```python
pandas, numpy, matplotlib, seaborn
scikit-learn, imbalanced-learn (SMOTE)
xgboost, lightgbm
