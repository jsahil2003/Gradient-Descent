# Custom Linear Regression Implementations with Gradient Descent Variants

This repository contains implementations of linear regression using different gradient descent optimization techniques from scratch, compared against scikit-learn’s built-in models on the diabetes dataset.

## Files Included

- **`full_batch_gradient_descent.py`**  
  Implements linear regression using Batch Gradient Descent (BGD), updating weights using the full training dataset each epoch.

- **`mini_batch_gradient_descent.py`**  
  Implements linear regression using Mini-Batch Gradient Descent (MBGD), updating weights using small batches of training data for faster and more stable convergence.

- **`stochastic_gradient_descent.py`**  
  Implements linear regression using Stochastic Gradient Descent (SGD), updating weights one sample at a time, with optional learning rate scheduling.

## Dataset

- Uses the **diabetes dataset** from `sklearn.datasets` for training and evaluation.  
- Dataset is split into training and testing sets (80% training, 20% testing).

## Features

- Custom implementations of linear regression with different gradient descent methods.  
- Custom prediction functions included.  
- Performance comparison with scikit-learn’s `LinearRegression` and `SGDRegressor` using R² score.

## Usage

- Run the individual Python scripts.  
- Adjust hyperparameters such as learning rate, epochs, batch size, and learning rate schedule inside the classes.  
- Use `r2_score` to evaluate and compare model performance.

## Results

- Print learned coefficients and intercepts from custom models alongside sklearn models.  
- Display R² scores to evaluate predictive accuracy of each implementation.

## Requirements

- Python 3.x  
- numpy  
- pandas  
- scikit-learn
