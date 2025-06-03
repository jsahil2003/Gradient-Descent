import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# Load diabetes dataset as DataFrame
x,y = load_diabetes(return_X_y=True,as_frame=True)

# Split data into train and test sets (80-20 split)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

# Initialize and train sklearn LinearRegression model
lr = LinearRegression()
lr.fit(x_train,y_train)

# Predict using sklearn model on test set
y_pred = lr.predict(x_test)


# Define custom Linear Regression using mini-batch gradient descent (MBGD)
class myMBGD:

    def __init__(self,batch_size = 25,rate=0.1,epochs=100 , lr_schedule = 0 ):
        self.intercept_ = None
        self.coef_ = None
        self.rate_ = rate
        self.init_rate_ = rate
        self.epochs_ = epochs
        self.batch_size_ = batch_size
        if not (0<= lr_schedule <1):
            raise ValueError("Learning schedule should be between 0 and 1")
        self.lr_schedule_ = lr_schedule

    def fit(self,x_train,y_train):
        n , num_of_coef = x_train.shape
        self.coef_ = np.ones(num_of_coef)
        self.intercept_ = 0
        
        for i in range(self.epochs_):
            # Update learning rate based on schedule
            self.rate_ = self.init_rate_ * (1 - self.lr_schedule_ * (i / self.epochs_))
            
            # Shuffle training data at start of each epoch
            shuffled_indices = np.random.permutation(len(x_train))
            x_train = x_train.iloc[shuffled_indices].reset_index(drop=True)
            y_train = y_train.iloc[shuffled_indices].reset_index(drop=True)
                
            # Loop over mini-batches
            for j in range(0, n ,self.batch_size_):
                x_train_temp = x_train.iloc[j : j + self.batch_size_]
                y_train_temp = y_train.iloc[j : j + self.batch_size_]
                divisor = x_train_temp.shape[0]
                
                # Calculate predictions for batch
                y_hat = np.dot(x_train_temp,self.coef_.T) + self.intercept_
                
                # Compute gradients
                intercept_der = -(2/divisor) * np.sum(y_train_temp-y_hat)
                self.intercept_ = self.intercept_ - (self.rate_*(intercept_der))
    
                coef_der = -(2/divisor) * np.dot((y_train_temp-y_hat).T,x_train_temp)
                self.coef_ = self.coef_ - (self.rate_ * coef_der)

    # Predict method using learned coefficients
    def predict(self,x_test):
        return np.dot(x_test,self.coef_.T) + self.intercept_


# Initialize custom MBGD model with specific hyperparameters
mymbgd = myMBGD(epochs = 150 , rate = 0.5 , batch_size = 30)

# Train custom model on training data
mymbgd.fit(x_train,y_train)

# Print learned coefficients and intercept from custom model
print(mymbgd.coef_,mymbgd.intercept_)

# Print sklearn model coefficients and intercept for comparison
print(lr.coef_ , lr.intercept_)

# Predict on test data using custom MBGD model
y_pred_mymbgd = mymbgd.predict(x_test)

# Calculate R2 score for custom model predictions
r2_score(y_test,y_pred_mymbgd)

# Calculate R2 score for sklearn model predictions
r2_score(y_test,y_pred)
