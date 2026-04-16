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
- Implemented custom functions for random and time-based data splits into train/validation/test sets.
- Built K-Fold, Grouped K-Fold, Stratified K-Fold, and Time Series cross-validation from scratch and compared them with `sklearn` equivalents.
- Applied Lasso regularization, correlation filtering, permutation importance, and SHAP to identify the most informative features.
- Compared Grid Search, Randomized Search, and Optuna-based Bayesian optimization for tuning ElasticNet hyperparameters.
- Evaluated all methods in terms of speed, metric stability, and overall model quality.

## Contents
- `data/` — raw datasets.
- `validation.ipynb` — Jupyter Notebook with completed tasks and conclusions.
