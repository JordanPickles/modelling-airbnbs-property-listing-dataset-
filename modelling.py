import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tabular_data import load_airbnb
from math import sqrt
import itertools
from itertools import product
import joblib
import json
import os



def split_data(X, y):
    """This function splits the data into a train, test and validate samples at a rate of 70%, 15% and 15% resepctively.
    Input: 
        tuples contatining features and labels in numerical form
    Output: 
        3 datasets containing tuples of features and labels from the original dataframe"""
    print(X)
    print(y)
    
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

def normalise_data(): #TODO Does this data need normalising before being used in the model?
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

    validation_mse = mean_squared_error(y_validation, y_validation_pred)
    validation_rmse = sqrt(mean_squared_error(y_validation, y_validation_pred))
    validation_r2 = r2_score(y_validation, y_validation_pred)

    print(f"Train MSE: {train_mse} | Train RMSE: {train_rmse} | Train R2: {train_r2}")
    print(f"Test MSE: {test_mse} | Test RMSE: {test_rmse} | Test R2 {test_r2}")
    print(f"Test MSE: {validation_mse} | Test RMSE: {validation_rmse} | Test R2 {validation_r2}")

    return validation_mse, validation_rmse, validation_r2
    
def custom_tune_regression_model_hyperparameters(model_class, X_train, y_train, X_test, y_test, X_validation, y_validation, hyperparameters):
    """This function is used to tune the hyperparameters of a regression model by iterating through all possible permutations of the input hyperparameters. The function takes in a model class, training, testing and validation data, as well as a dictionary of hyperparameters and their possible values. The function returns the best model, the best set of hyperparameters and a dictionary of performance metrics.

    Inputs:
        - model_class (class): a model class to be used in the function
        - X_train (np.ndarray): a numpy array of feature values for training
        - y_train (np.ndarray): a numpy array of label values for training
        - X_test (np.ndarray): a numpy array of feature values for testing
        - y_test (np.ndarray): a numpy array of label values for testing
        - X_validation (np.ndarray): a numpy array of feature values for validation
        - y_validation (np.ndarray): a numpy array of label values for validation
        - hyperparameters (dict): a dictionary of hyperparameters and their possible values

    Returns:
        - best_model : best model
        - best_hyperparameters (dict): a dictionary containing the best hyperparameters
        - performance_metrics (dict): a dictionary containing the performance metrics

    """

    best_model = None
    best_hyperparameters = {}
    performance_metrics = {} 
    best_validation_rmse = float('inf')

    # Provides all possible permutations of hyperparameter combinations
    keys = hyperparameters.keys()
    values = hyperparameters.values()
    hyperparameter_combinations = [dict(zip(keys, combination)) for combination in product(*values)]

    # Iterates through hyperparameter combinations to decifer the optimal hyperparameters
    for params in hyperparameter_combinations:
        model = model_class(**params)
        model.fit(X_train, y_train)
        validation_predictions = model.predict(X_validation)
        validation_rmse = sqrt(mean_squared_error(validation_predictions, y_validation))

        if validation_rmse < best_validation_rmse:
            best_model = model
            best_hyperparameters = params
            best_val_rmse = validation_rmse

    # Provides test metrics
    y_test_pred = best_model.predict(X_test)
    test_rmse = sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # Map metrics to the performance metrics  
    performance_metrics['validation_rmse'] = validation_rmse
    performance_metrics['test_rmse'] = test_rmse
    performance_metrics['test_r_squared'] = test_r2
    performance_metrics['test_mae'] = test_mae

    
    return best_model, best_hyperparameters, performance_metrics

def tune_regression_model_hyperparameters(model_class, X_train, y_train, X_test, y_test, X_validation, y_validation, hyperparameters):
    """
    This function takes in a regression model class, training, testing and validation datasets, and a dictionary of hyperparameters to tune. It then uses GridSearchCV to find the best hyperparameters for the model, and returns the best model, the best hyperparameters, and performance metrics (validation and test rmse, r2, and mae).

    Inputs:
        model_class: a regression model class (e.g. RandomForestRegressor)
        X_train: training set of features
        y_train: training set of labels
        X_test: testing set of features
        y_test: testing set of labels
        X_validation: validation set of features
        y_validation: validation set of labels
        hyperparameters: dictionary of hyperparameters to tune
    Outputs:
        best_model: an instance of the model_class, with the best hyperparameters found
        best_hyperparameters: a dictionary of the best hyperparameters found
        performance_metrics: a dictionary of performance metrics (validation and test rmse, r2, and mae)
    """

    performance_metrics = {}
    
    grid_search = GridSearchCV(model_class, hyperparameters, scoring = 'neg_mean_squared_error', cv = 5) 
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_hyperparameters = grid_search.best_params_

    # Provides Validation Metrics
    y_validation_pred = best_model.predict(X_validation)
    validation_rmse = sqrt(mean_squared_error(y_validation, y_validation_pred))
    validation_r2 = r2_score(y_validation, y_validation_pred)
    validation_mae = mean_absolute_error(y_validation, y_validation_pred)

    # Map metrics to the performance metrics 
    performance_metrics['validation_rmse'] = validation_rmse
    performance_metrics['validation_r2'] = validation_r2
    performance_metrics['validation_mae'] = validation_mae
    
    return best_model, best_hyperparameters, performance_metrics

def save_model(model, hyperparameters, metrics, model_folder):
    """This function saves a trained model, its associated hyperparameters and performance metrics to a specified folder.
    Inputs:
        model: A trained machine learning model
        hyperparameters: A dictionary of the best hyperparameters used to train the model
        metrics: A dictionary of the performance metrics of the model on test and validation sets
        model_folder: A string specifying the directory path where the model and associated files will be saved."""

    
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    joblib.dump(model, f'{model_folder}/model.joblib')
    with open(f'{model_folder}/hyperparameters.json', 'w') as f:
        json.dump(hyperparameters, f)
    with open(f'{model_folder}/metrics.json', 'w') as f:
        json.dump(metrics, f)

def evaluate_all_models(X_train, y_train, X_test, y_test, X_validation, y_validation): 
    """This function evaluate different regression models by tuning the hyperparameters and then saving the best models, hyperparameters and performance metrics to specific folder.

    Inputs:
        X_train: A numpy array or pandas dataframe, representing the training set for the independent variables.
        y_train: A numpy array or pandas dataframe, representing the training set for the dependent variable.
        X_test: A numpy array or pandas dataframe, representing the testing set for the independent variables.
        y_test: A numpy array or pandas dataframe, representing the testing set for the dependent variable.
        X_validation: A numpy array or pandas dataframe, representing the validation set for the independent variables.
        y_validation: A numpy array or pandas dataframe, representing the validation set for the dependent variable.
    
    Outputs:
        It saves the best models, hyperparameters and performance metrics of all evaluated models to specific folder."""

    sgd_hyperparameters = {
        'penalty': ['l2', 'l1','elasticnet'],
        'alpha': [0.1, 0.01, 0.001, 0.0001],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9 ],
        'max_iter': [500, 1000, 1500, 2000],
        'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive']
    }

    decision_tree_hyperparameters = {
    'max_depth': [10, 20, 50],
    'min_samples_split': [2, 4, 6, 8],
    'min_samples_leaf': [1, 3, 5, 7],
    'splitter': ['best', 'random'] 
    }
    random_forest_hyperparameters = {
        'n_estimators': [50, 100, 150],
        'max_depth': [10,20,50],
        'min_samples_split': [2, 4, 6, 8],
        'min_samples_leaf': [1, 3, 5, 7]

    }
    gradient_boost_hyperparameters = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.1, 0.001, 0.0001],
        'criterion': ['friedman_mse', 'squared_error'],
        'min_samples_split': [2, 4, 6, 8],
        'min_samples_leaf': [1, 3, 5, 7]
    }
    model_hyperparameters = [decision_tree_hyperparameters, random_forest_hyperparameters, gradient_boost_hyperparameters]

    models_dict = {
        'SGD Regressor': [SGDRegressor(), sgd_hyperparameters],
        'Decision Tree Regressor': [DecisionTreeRegressor(), decision_tree_hyperparameters],
        'Random Forest Regressor': [RandomForestRegressor(), random_forest_hyperparameters],
        'Gradient Boosting Regressor': [GradientBoostingRegressor(), gradient_boost_hyperparameters]

    }
    
    # Create required directories to save the models to   
    if not os.path.exists('./models'):
        os.makedirs('./models')
    if not os.path.exists('./models/regression'):
        os.makedirs('./models/regression')
    if not os.path.exists('./models/regression/linear_regression'):
        os.makedirs('./models/regression/linear_regression')

    # For loop iterates through the models provided and calls the tune_regression_mode_hyperparameters
    for key, values in models_dict.items(): #TODO - should a random seed be included here?
        model, hyperparameters = values
        best_model, best_hyperparameters, performance_metrics = tune_regression_model_hyperparameters(model, X_train, y_train, X_test, y_test, X_validation, y_validation, hyperparameters)
        folder_path = f'./models/regression/linear_regression/{key}'
        save_model(best_model, best_hyperparameters, performance_metrics, folder_path) 
        

def find_best_model():
    """This function compares the Root Mean Squared Error (RMSE) of the trained models on validation set and returns the model with the lowest RMSE.
    Inputs:
        None
    Outputs:
        Prints the model name with the lowest RMSE
    """
    models = ['SGD Regressor', 'Decision Tree Regressor', 'Random Forest Regressor', 'Gradient Boosting Regressor']
    best_model = None
    best_rmse = float('inf')
    best_r2 = 0
    for model in models:
        with open(f'./models/regression/linear_regression/{model}/metrics.json') as f: 
            metrics = json.load(f)
            validation_r2 = metrics['validation_r2']
            validation_rmse = metrics['validation_rmse']
            validation_mae = metrics['validation_mae']
            print(f'{model}: RMSE: {validation_rmse}')

            if validation_rmse < best_rmse:
                best_rsme = validation_rmse
                best_model = model

    
    return print(f'The model with the lowest RMSE is: {best_model}')

if __name__ == "__main__":
    X, y = load_airbnb(pd.read_csv('./airbnb-property-listings/tabular_data/clean_tabular_data.csv'), 'Price_Night')
    X_train, y_train, X_test, y_test, X_validation, y_validation = split_data(X, y) 
    evaluate_all_models(X_train, y_train, X_test, y_test, X_validation, y_validation)
    find_best_model()