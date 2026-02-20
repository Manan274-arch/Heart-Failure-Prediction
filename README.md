# Heart Failure Mortality Prediction — Comparative Machine Learning Study

## Overview

This project performs a comparative analysis of four machine learning models for predicting mortality in heart failure patients using clinical data.

The objective is to evaluate different model families and identify the most effective approach for structured tabular medical data.

---

## Dataset

- Dataset Source: Kaggle - Heart Failure Clinical Records Dataset

- 299 patient records
- 12 clinical features
- Target variable: `DEATH_EVENT` (0 = Survived, 1 = Died)

The dataset includes demographic, biochemical, and clinical measurements such as:

- Age
- Ejection fraction
- Serum creatinine
- Serum sodium
- Diabetes
- Anaemia
- Smoking status
- Follow-up time

---

## Preprocessing

- Stratified train-test split (80-20)
- Log transformation for skewed variables:
  - Creatinine phosphokinase
  - Serum creatinine
- Standard scaling for continuous variables (for linear & neural models)
- Class imbalance handling using balanced class weights

---

## Models Compared

1. Logistic Regression  
2. Decision Tree  
3. Random Forest  
4. Neural Network (TensorFlow/Keras)

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC (primary metric)

---

## Results

| Model | Accuracy | F1 Score | ROC-AUC |
|--------|----------|----------|---------|
| Logistic Regression | 0.783 | 0.629 | 0.858 |
| Decision Tree (Tuned) | 0.783 | 0.683 | 0.845 |
| Random Forest | **0.833** | **0.706** | **0.910** |
| Neural Network | 0.767 | 0.650 | 0.852 |

---

## Conclusion

Random Forest achieved the highest ROC-AUC (0.91) and overall performance.  
This confirms that tree ensemble methods are highly effective for small structured tabular datasets.

Neural networks did not outperform ensemble methods due to the limited dataset size.

---
