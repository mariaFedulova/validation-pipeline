# Model Quality Estimation
This work explores proper validation techniques for machine learning algorithms. It covers data splitting strategies, prevention of information leakage, hyperparameter tuning, and feature selection methods.

## Overview
This project explores validation strategies and model tuning techniques for ensuring reliable machine learning performance and preventing data leakage.

- Implementation of data splitting methods (random and time-based splits)
- Custom cross-validation schemes (K-Fold, Grouped K-Fold, Stratified K-Fold, Time Series Split)
- Feature selection techniques (Lasso coefficients, correlation filtering, permutation importance, SHAP)
- Hyperparameter optimization (Grid Search, Randomized Search, Optuna Bayesian tuning)
- Comparison with scikit-learn implementations
- Evaluation of methods by speed, metric stability, and model quality

## Results
Custom implementations of all splitting and cross-validation methods matched scikit-learn outputs exactly, confirming correctness. Feature selection and hyperparameter tuning experiments demonstrated that Optuna achieved optimal ElasticNet performance with fewer iterations. Time series cross-validation provided more realistic performance estimates than standard K-Fold due to temporal structure in the data.

## Contents
- `data/` - datasets.
- `validation.ipynb` - Jupyter Notebook with completed tasks and conclusions.
