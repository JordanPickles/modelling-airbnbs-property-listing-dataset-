import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from tabular_data import load_airbnb
from math import sqrt
from itertools import product



def split_data(X, y):
    """This function splits the data into a train, test and validate samples at a rate of 70%, 15% and 15% resepctively.
    Input: tuples contatining features and labels in numerical form
    Output: 3 datasets containing tuples of features and labels from the original dataframe"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    print("Number of samples in:")
    print(f"    Training: {len(y_train)}")
    print(f"    Testing: {len(y_test)}")

    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size = 0.5)

    print("Number of samples in:")
    print(f"    Training: {len(y_train)}")
    print(f"    Testing: {len(y_test)}")
    print(f"    Validation: {len(y_validation)}")

    return X_train, y_train, X_test, y_test, X_validation, y_validation

def normalise_data(): #TODO normalise the data - find best practice
    pass

def train_model(X_train, y_train, X_test, y_test, X_validation, y_validation):
    """This function fits the train data to a SGD regression model tests the error level of the model on the train, test and validation datasets
    Input: 
        X_train, X_test and X_validation consists of the features being passed into the model
        y_train, y_test and y_validation consists of the labels being passed into the model

    Output: The mean squared error of the model on the train, test and validation sets"""
    
    model = SGDRegressor()
    np.random.seed(2)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_validation_pred = model.predict(X_validation)
    y_test_pred = model.predict(X_test)

    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = mean_squared_error(y_train, y_train_pred, squared = False)
    train_r2 = r2_score(y_train, y_train_pred)
    
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"Train MSE: {train_mse} | Train RMSE: {train_rmse} | Train R2: {train_r2}")
    print(f"Test MSE: {test_mse} | Test RMSE: {test_rmse} | Test R2 {test_r2}")

    
def custom_tune_regression_model_hyperparameters(model_class, X_train, y_train, X_test, y_test, X_validation, y_validation, hyperparameters):
    best_model = []
    best_hyperparameters = {}
    performance_metrics = {} 
    best_val_rmse = float('inf')

    # Provides all possible permutations of hyperparameter combinations
    keys = hyperparameters.keys()
    values = hyperparameters.values()
    hyperparameter_combinations = [dict(zip(keys, combination)) for combination in product(*values)]

    # Iterates through hyperparameter combinations to decifer the optimal hyperparameters
    for params in hyperparameter_combinations:
        model = model_class(**params)
        model.fit(X_train, y_train)
        val_predictions = model.predict(X_validation)
        val_rmse = sqrt(mean_squared_error(val_predictions, y_validation))

        if val_rmse < best_val_rmse:
            best_model = model
            best_params = params
            best_val_rmse = val_rmse

    # Provides test metrics
    y_test_pred = best_model.predict(X_test)
    test_rmse = sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # Map metrics to the performance metrics 
    performance_metrics['validation_rmse'] = best_val_rmse
    performance_metrics['test_rmse'] = test_rmse
    performance_metrics['r_squared'] = test_r2
    performance_metrics['test'] = test_mae

    
    return best_model, best_hyperparameters, performance_metrics

def tune_regression_model_hyperparameters(model_class, X_train, y_train, X_test, y_test, X_validation, y_validation, hyperparameters):
    

if __name__ == "__main__":
    X, y = load_airbnb(pd.read_csv('./airbnb-property-listings/tabular_data/clean_tabular_data.csv'), 'Price_Night')
    X_train, y_train, X_test, y_test, X_validation, y_validation = split_data(X, y)
    train_model(X_train, y_train, X_test, y_test, X_validation, y_validation)
