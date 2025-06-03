import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import r2_score

# Load diabetes dataset as numpy arrays
x,y = load_diabetes(return_X_y=True)

# Split data into training and testing sets (80-20 split)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

# Initialize and train sklearn LinearRegression model
lr = LinearRegression()
lr.fit(x_train,y_train)

# Initialize and train sklearn SGDRegressor with constant learning rate
sgd = SGDRegressor(max_iter=100, eta0=0.05, learning_rate='constant')
sgd.fit(x_train,y_train)

# Predict using sklearn SGDRegressor
y_pred_sgd = sgd.predict(x_test)

# Predict using sklearn LinearRegression
y_pred = lr.predict(x_test)


# Custom Linear Regression using Stochastic Gradient Descent (SGD)
class mySGDRegressor:

    def __init__(self, rate=0.1, epochs=100, lr_schedule=0):
        self.intercept_ = None
        self.coef_ = None
        self.rate_ = rate          # current learning rate
        self.init_rate_ = rate     # initial learning rate for scheduling
        self.epochs_ = epochs
        if not (0 <= lr_schedule < 1):
            raise ValueError("Learning schedule should be between 0 and 1")
        self.lr_schedule_ = lr_schedule

    def fit(self, x_train, y_train):
        num_of_coef = x_train.shape[1]
        n_rows = x_train.shape[0]
        self.coef_ = np.ones(num_of_coef)
        self.intercept_ = 0
        
        # Training loop over epochs
        for i in range(self.epochs_):
            # Update learning rate according to schedule (linear decay)
            self.rate_ = self.init_rate_ * (1 - self.lr_schedule_ * (i / self.epochs_))
            
            # Stochastic updates: random sample at each step
            for j in range(n_rows):
                idx = np.random.randint(0, n_rows)  # randomly pick one sample
                
                y_hat = np.dot(x_train[idx], self.coef_.T) + self.intercept_
                
                # Compute gradient for intercept
                intercept_der = -2 * (y_train[idx] - y_hat)
                self.intercept_ = self.intercept_ - (self.rate_ * intercept_der)

                # Compute gradient for coefficients
                coef_der = -2 * np.dot((y_train[idx] - y_hat).T, x_train[idx])
                self.coef_ = self.coef_ - (self.rate_ * coef_der)

    # Prediction using learned coefficients and intercept
    def predict(self, x_test):
        return np.dot(x_test, self.coef_.T) + self.intercept_


# Initialize custom SGD model with given parameters
mysgd = mySGDRegressor(epochs=100, rate=0.05, lr_schedule=0.05)

# Train the custom SGD model
mysgd.fit(x_train, y_train)

# Print learned coefficients and intercept from custom SGD model
print(mysgd.coef_, mysgd.intercept_)

# Print sklearn SGDRegressor coefficients and intercept
print(sgd.coef_, sgd.intercept_)

# Print sklearn LinearRegression coefficients and intercept
print(lr.coef_, lr.intercept_)

# Predict on test data using custom SGD model
y_pred_mysgd = mysgd.predict(x_test)

# Calculate R2 score for custom SGD predictions
r2_score(y_test, y_pred_mysgd)

# Calculate R2 score for sklearn SGD predictions
r2_score(y_test, y_pred_sgd)

# Calculate R2 score for sklearn LinearRegression predictions
r2_score(y_test, y_pred)
