# Problem Statment
 The objective of this project is to perform binary classification on the Adult Income dataset to predict whether an individual earns more than $50,000 per year based on demographic and employment-related attributes. Multiple machine learning models are implemented, evaluated, and compared. The final solution is deployed as an interactive Streamlit web application.

# Dataset Description:
The Adult Income dataset is sourced from the UCI Machine Learning Repository. It contains demographic and employment-related information about individuals.

Total Instances: ~48,000
Total Features: 14 input features
Target Variable: Income (<=50K or >50K)
Type: Binary Classification

Key Features Include:
Age
Workclass
Education
Marital Status
Occupation
Relationship
Race
Sex
Capital Gain
Capital Loss
Hours per Week
Native Country

The dataset was preprocessed by:
Handling missing values
Encoding categorical variables
Feature scaling (where required)

# Model Comparison

| Model               | Accuracy | AUC     | Precision | Recall  | F1 Score | MCC     |
|---------------------|----------|---------|-----------|---------|----------|---------|
| Logistic Regression | 0.8279   | 0.8608  | 0.7246    | 0.4598  | 0.5626   | 0.4806  |
| Decision Tree       | 0.8138   | 0.7555  | 0.6067    | 0.6435  | 0.6246   | 0.5013  |
| KNN                 | 0.8340   | 0.8569  | 0.6711    | 0.6091  | 0.6386   | 0.5322  |
| Naive Bayes         | 0.8081   | 0.8644  | 0.7044    | 0.3495  | 0.4672   | 0.3994  |
| Random Forest       | 0.8597   | 0.9101  | 0.7448    | 0.6346  | 0.6853   | 0.5989  |
| XGBoost             | 0.8741   | 0.9288  | 0.7722    | 0.6767  | 0.7213   | 0.6427  |


## Model Performance Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Performs reasonably well as a baseline model with good AUC, but recall is relatively low, indicating difficulty in identifying positive-income cases. |
| Decision Tree | Shows balanced precision and recall, but overall accuracy is slightly lower due to overfitting tendencies of single decision trees. |
| kNN | Provides stable performance with balanced metrics, though computationally expensive and sensitive to feature scaling. |
| Naive Bayes | Achieves good AUC but very low recall, suggesting the independence assumption limits performance on this dataset. |
| Random Forest (Ensemble) | Significantly improves performance over individual models by reducing variance and overfitting, achieving strong accuracy and F1 score. |
| XGBoost (Ensemble) | Best-performing model overall, with highest accuracy, AUC, F1 score, and MCC, demonstrating the effectiveness of gradient boosting on structured tabular data. |