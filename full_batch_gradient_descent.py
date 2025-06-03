import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load diabetes dataset as numpy arrays
x,y = load_diabetes(return_X_y=True)

# Split data into train and test sets (80-20 split)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

# Initialize and train sklearn LinearRegression model
lr = LinearRegression()
lr.fit(x_train,y_train)

# Predict on test data using sklearn model
y_pred = lr.predict(x_test)


# Define custom Linear Regression using full-batch Gradient Descent
class myLRGD:

    def __init__(self,rate=0.1,epochs=100):
        self.intercept_ = None
        self.coef_ = None
        self.rate_ = rate
        self.epochs_ = epochs

    def fit(self,x_train,y_train):
        num_of_coef = x_train.shape[1]
        n = x_train.shape[0]
        self.coef_ = np.ones(num_of_coef)
        self.intercept_ = 0
        
        # Run gradient descent for given epochs
        for i in range(self.epochs_):
            # Predict current values
            y_hat = np.dot(x_train,self.coef_.T) + self.intercept_
            
            # Calculate gradient for intercept
            intercept_der = -(2/n) * np.sum(y_train-y_hat)
            self.intercept_ = self.intercept_ - (self.rate_*(intercept_der))

            # Calculate gradient for coefficients
            coef_der = -(2/n) * np.dot((y_train-y_hat).T,x_train)
            self.coef_ = self.coef_ - (self.rate_ * coef_der)

    # Predict method using learned parameters
    def predict(self,x_test):
        return np.dot(x_test,self.coef_.T) + self.intercept_


# Initialize custom gradient descent model with hyperparameters
mylrgd = myLRGD(epochs = 5000 , rate = 0.8)

# Train custom model on training data
mylrgd.fit(x_train,y_train)

# Print learned coefficients and intercept from custom model
print(mylrgd.coef_,mylrgd.intercept_)

# Print sklearn model coefficients and intercept for comparison
print(lr.coef_ , lr.intercept_)

# Predict on test data using custom gradient descent model
y_pred_mylr = mylrgd.predict(x_test)

# Calculate R2 score for custom model predictions
r2_score(y_test,y_pred_mylr)

# Calculate R2 score for sklearn model predictions
r2_score(y_test,y_pred)
