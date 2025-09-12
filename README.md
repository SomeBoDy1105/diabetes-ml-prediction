# diabetes-ml-prediction
# 🩺 Diabetes Prediction using Machine Learning

This repository contains a complete machine learning pipeline for predicting diabetes based on clinical features such as glucose level, BMI, age, and insulin measurements.  
The project benchmarks three models — **Logistic Regression, Random Forest, and SVM (with GridSearchCV tuning)** — and compares their performance on accuracy, precision, recall, F1 score, and ROC AUC.

---

## 📊 Project Overview
- **Dataset**:Diabetes Dataset (768 samples, 8 features + outcome).
- **Goal**: Early diabetes prediction to support clinical decision-making.
- **Models Implemented**:
  - Logistic Regression (baseline, interpretable model)
  - Random Forest (ensemble trees, hyperparameter tuning with GridSearchCV)
  - Support Vector Machine (SVM, RBF kernel with tuned `C` and `gamma`)
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1 score, ROC AUC.
- **Best Model**:
  - Random Forest achieved the highest accuracy (0.74) and F1 score (0.61).
  - SVM achieved the best ROC AUC (0.817).

---

## 📂 Repository Structure

diabetes-ml-prediction/
│── diabetes_project.ipynb # 
│── diabetes.csv #  
│── report.pdf # 
│── README.md # 

yaml
Copier le code

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

git clone https://github.com/fouad3966/diabetes-ml-prediction.git
cd diabetes-ml-prediction
2️⃣ Run the Notebook
Open Google Colab, upload diabetes_project.ipynb, and run all cells.
You can upload your own dataset or use the included diabetes.csv.

📈 Results
Model	Accuracy	Precision	Recall	F1	ROC AUC
Logistic Regression	0.71	0.60	0.50	0.55	0.813
Random Forest	0.74	0.65	0.57	0.61	0.808
SVM (RBF, tuned)	0.69	0.58	0.48	0.53	0.817

Random Forest → Best practical classifier (highest accuracy and F1).

SVM → Strongest ranking ability (highest ROC AUC).

NOTE: For more details read the Report attached

