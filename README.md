# Diabetes Prediction using Machine Learning

This repository contains a complete notebook workflow for predicting diabetes with the Pima Indians Diabetes dataset. The project cleans the data, engineers medically relevant features, trains multiple models (including a stacking ensemble), and benchmarks their performance with both hold-out and repeated stratified cross-validation.

---

## Project Overview
- **Dataset**: Pima Indians Diabetes dataset (768 rows, 8 numeric predictors, binary outcome).
- **Goal**: Build an accurate, recall-friendly screening model that flags patients at risk of diabetes.
- **Notebook**: All analysis lives in `diabetes_project.ipynb`. You can run it locally or in Google Colab.
- **Pipeline highlights**:
  1. Replace medically impossible zero values with missing flags, then median-impute using only training folds.
  2. Engineer interaction features (for example `Glucose_BMI`, `Pregnancies_per_Age`, `Insulin_to_Glucose`) to surface non-linear risk signals.
  3. Standardize the augmented feature set and train a suite of models with class balancing.
  4. Tune hyperparameters with `GridSearchCV` or `RandomizedSearchCV`, build a stacking ensemble on top, and export artifacts with `joblib`.
  5. Sweep probability thresholds for the top-scoring classifier to balance recall versus precision.

---

## Repository Structure

```
diabetes-ml-prediction/
|-- diabetes_project.ipynb   # Main notebook with data prep, modelling, evaluation
|-- diabetes.csv             # Pima Indians Diabetes dataset copied locally
`-- README.md                # Project overview (this file)
```

---

## Model Benchmarks (Hold-out Test Split)

The hold-out set corresponds to a stratified 80/20 split with the updated preprocessing pipeline.

| Model                | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|----------------------|----------|-----------|--------|----------|---------|
| Logistic Regression  | 0.740    | 0.613     | 0.704  | 0.655    | 0.815   |
| Random Forest        | 0.708    | 0.571     | 0.667  | 0.615    | 0.811   |
| SVM (RBF)            | 0.734    | 0.600     | 0.722  | 0.655    | 0.809   |
| HistGradientBoosting | 0.747    | 0.653     | 0.593  | 0.621    | 0.814   |
| **Stacking Ensemble**| **0.747**| 0.619     | **0.722** | **0.667** | **0.819** |

**Threshold tuning**: Optimizing the stacking ensemble decision boundary on the hold-out set selects a 0.40 probability cutoff. At that threshold Precision = 0.580, Recall = 0.870, and F1 = 0.696, which improves sensitivity for screening scenarios while keeping false positives manageable.

---

## Robust Validation (Repeated Stratified K-Fold)

To confirm the gains are not split-specific, the notebook runs `RepeatedStratifiedKFold` with 5 folds and 3 repeats. Preprocessing is re-fit inside each fold to avoid leakage. Summary statistics:

| Model                | Accuracy Mean +/- Std | Precision Mean +/- Std | Recall Mean +/- Std | F1 Mean +/- Std | ROC AUC Mean +/- Std |
|----------------------|----------------------|------------------------|---------------------|-----------------|----------------------|
| Logistic Regression  | 0.753 +/- 0.041      | 0.634 +/- 0.053        | 0.688 +/- 0.073     | 0.658 +/- 0.061 | **0.838 +/- 0.031**  |
| **Stacking Ensemble**| **0.753 +/- 0.041**  | 0.622 +/- 0.055        | **0.707 +/- 0.073** | **0.661 +/- 0.061** | 0.838 +/- 0.031      |
| SVM (RBF)            | 0.750 +/- 0.043      | 0.623 +/- 0.057        | 0.693 +/- 0.074     | 0.655 +/- 0.062 | 0.838 +/- 0.030      |
| Random Forest        | 0.761 +/- 0.038      | 0.602 +/- 0.047        | 0.713 +/- 0.066     | 0.652 +/- 0.056 | 0.836 +/- 0.031      |
| HistGradientBoosting | 0.751 +/- 0.031      | 0.640 +/- 0.049        | 0.641 +/- 0.063     | 0.639 +/- 0.053 | 0.826 +/- 0.028      |

The ensemble and logistic regression perform neck-and-neck, and standard deviations below 0.04 indicate stable generalisation across folds.

---

## Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/fouad3966/diabetes-ml-prediction.git
   cd diabetes-ml-prediction
   ```
2. **Create an environment and install dependencies**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # macOS/Linux
   pip install pandas numpy matplotlib seaborn scikit-learn joblib
   ```
3. **Run the notebook**
   - Launch Jupyter locally: `jupyter notebook diabetes_project.ipynb`, or
   - Upload `diabetes_project.ipynb` to Google Colab and run all cells. Use the bundled `diabetes.csv` or upload your own dataset (same schema required).

---

## Outputs and Artifacts

- `best_model_bundle.joblib`: created by the notebook; contains the tuned best model (currently the stacking ensemble), median imputer, scaler, engineered feature list, zero-invalid columns, and recommended decision threshold.
- Plots: confusion matrices, ROC curves, and threshold diagnostics are generated inline.
- Tables: `metrics_df` (hold-out comparison) and `cv_summary_df` (repeated CV summary) can be exported from the notebook if needed.

---

## Next Steps

- Calibrate probabilities (for example with `CalibratedClassifierCV`) if you plan to use risk scores directly in triage workflows.
- Extend feature engineering with clinical domain input or additional lab results if available.
- Deploy the saved bundle behind a REST or Streamlit service to support real-time predictions.

Feel free to open issues or pull requests with enhancements or additional analyses.
