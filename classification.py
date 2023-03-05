import pandas as pd
import numpy as np
from tabular_data import load_airbnb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import os
import joblib
import json

def split_data(X, y):
    """Split data into train, test, and validation sets, with normalization.

    Parameters:
        X (Matrix): Features
        y (Vector): Labels

    Returns:
        Normalized X_train, y_train, X_test, y_test, X_validation, y_validation.    """

    le = LabelEncoder()

    #Encodes the labels
    y_encoded = le.fit_transform(y)


    #Splits the data into train and test data at 70% and 30% respectively
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size = 0.3)

    #Splits the test data into test and validation set at 15% each
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size = 0.5)

    #Normalises the features using MinMaxScaler
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    X_validation = scaler.fit_transform(X_validation)

    return X_train, y_train, X_test, y_test, X_validation, y_validation


def classification_model(X_train, y_train, X_test, y_test, X_validation, y_validation):
    """Train and evaluate a logistic regression model on the input train and test data.

    Parameters:
    X_train (Matrix): Normalized features for training
    y_train (Vector): Lables for training
    X_test (Matrix): Normalized features for testing
    y_test (Vector): Lables for testing
    X_validation (Matrix): Normalized features for validation
    y_validation (Vector): Labels for validation

    Returns:
    None: The function prints evaluation metrics for the logistic regression model on the train and test data.
    """

    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Train set evaluation metrics
    print(f'Train Accuracy Score: {accuracy_score(y_train, y_train_pred)}')
    print(f'Train Precision Score: {precision_score(y_train, y_train_pred, average="micro")}')
    print(f'Train Recall Score: {recall_score(y_train, y_train_pred, average="micro")}')    
    print(f'Train F1 Score: {f1_score(y_train, y_train_pred, average="micro")}')

    # Test set evaluation metrics
    print(f'Test Accuracy Score: {accuracy_score(y_test, y_test_pred)}')
    print(f'Test Precision Score: {precision_score(y_test, y_test_pred, average="micro")}')
    print(f'Test Recall Score: {recall_score(y_test, y_test_pred, average="micro")}')    
    print(f'Test F1 Score: {f1_score(y_test, y_test_pred, average="micro")}')


def tune_classification_model_hyperparameters(model_class, X_train, y_train, X_test, y_test, X_validation, y_validation, hyperparameters):
    """
    This function performs hyperparameter tuning for a classification model and returns the best model, its hyperparameters, and its performance metrics on a validation set.
    
    Parameters:
    model_class (class): A class for a scikit-learn classifier that implements `fit` and `predict` methods.
    X_train (Matrix): Normalized features for training
    y_train (Vector): Lables for training
    X_test (Matrix): Normalized features for testing
    y_test (Vector): Lables for testing
    X_validation (Matrix): Normalized features for validation
    y_validation (Vector): Labels for validation
    hyperparameters (dict): The hyperparameters to be tested by GridSearchCV.
    
    Returns:
    best_model (scikit-learn classifier instance): The best classifier instance after tuning hyperparameters.
    best_hyperparameters (dict): The best hyperparameters obtained from GridSearchCV.
    performance_metrics (dict): A dictionary of performance metrics of the best model on the validation set. The metrics include:
        - validation_accuracy (float): Accuracy score of the model on the validation set.
        - validation_precision (float): Precision score of the model on the validation set.
        - validation_recall (float): Recall score of the model on the validation set.
        - validation_f1_score (float): F1 score of the model on the validation set.
    """
    
    performance_metrics = {}
    
    grid_search = GridSearchCV(model_class, hyperparameters, scoring = 'f1_micro', cv = 5) 
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_hyperparameters = grid_search.best_params_

    # Provides Validation Metrics
    y_validation_pred = best_model.predict(X_validation)
    y_validation_accuracy = accuracy_score(y_validation, y_validation_pred)
    y_validation_precision = precision_score(y_validation, y_validation_pred, average='micro')
    y_validation_recall = recall_score(y_validation, y_validation_pred, average='micro')
    y_validation_f1 = f1_score(y_validation, y_validation_pred, average='micro')

    # Maps metrics to the performance metrics dict
    performance_metrics['validation_accuracy'] = y_validation_accuracy
    performance_metrics['validation_precision'] = y_validation_precision
    performance_metrics['validation_recall'] = y_validation_recall
    performance_metrics['validation_f1_score'] = y_validation_f1

    
    return best_model, best_hyperparameters, performance_metrics

def save_model(model, hyperparameters, metrics, model_folder):
    """This function saves a trained model, its associated hyperparameters and performance metrics to a specified folder.
    Parameters:
        model: Machine learning model name
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

    Parameters:
        X_train (Matrix): Normalized features for training
        y_train (Vector): Lables for training
        X_test (Matrix): Normalized features for testing
        y_test (Vector): Lables for testing
        X_validation (Matrix): Normalized features for validation
        y_validation (Vector): Labels for validation
    
    Outputs:
        It saves the best models, hyperparameters and performance metrics of all evaluated models to specific folder."""
    
    # Adds Hyperparameters for hyperparameter for each model
    logistic_regression_hyperparameters = {
        'penalty': ['l1', 'l2', 'elasticnet'],
        'max_iter': [100, 1000, 10000],
        'solver': ['lbfgs', 'newton-cg', 'sag', 'saga']
    }
    decision_tree_classifier_hyperparameters = {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'splitter': ['best', 'random'],
        'max_depth': [10, 20, 50], #TODO what is a good number?
        'min_samples_split': [2, 4, 6, 8],
        'min_samples_leaf': [1, 2, 3, 4]
    }
    random_forest_classifier_hyperparameters = {
        'n_estimators': [50, 100, 200],
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': [10, 20, 50],
        'min_samples_split': [2, 4, 6,8],
        'min_samples_leaf': [1, 2, 3, 4]
    }
    gradient_boosting_classifier_hyperparameters = {
        'criterion': ['friedman_mse', 'squared_error'],
        'min_samples_split': [2, 4, 6, 8],
        'min_samples_leaf': [1, 2, 3, 4],
        'max_depth': [2, 3, 4, 5]
    }
    
    # Adds models to a dict to be iterated through
    classification_models_dict = {
        'Logistic Regression': [LogisticRegression(),logistic_regression_hyperparameters], 
        'Decision Tree Classifier': [DecisionTreeClassifier() ,decision_tree_classifier_hyperparameters], 
        'Random Forest Classifier': [RandomForestClassifier(), random_forest_classifier_hyperparameters], 
        'Gradient Boosting Classifier': [GradientBoostingClassifier() ,gradient_boosting_classifier_hyperparameters]
    }
    
    # Create required directories to save the models to   
    if not os.path.exists('./models'):
        os.makedirs('./models')
    if not os.path.exists('./models/classification'):
        os.makedirs('./models/classification')
  

    # For loop iterates through the models provided and calls the tune_regression_mode_hyperparameters
    for key, values in classification_models_dict.items(): 
        model, hyperparameters = values
        best_model, best_hyperparameters, performance_metrics = tune_classification_model_hyperparameters(model, X_train, y_train, X_test, y_test, X_validation, y_validation, hyperparameters)
        folder_path = f'./models/classification/{key}'
        save_model(best_model, best_hyperparameters, performance_metrics, folder_path) 
        

def find_best_model(): 
    """This function compares the F1 error score of the trained models on validation set and returns the model with the lowest F1 score.
    Parameters:
        None
    Outputs:
        Prints the model name with the lowest F1 score
    """
    
    models = ['Logistic Regression', 'Decision Tree Classifier', 'Random Forest Classifier', 'Gradient Boosting Classifier']
    best_model = None
    best_f1_score= float('inf') 
    
    for model in models:
        with open(f'./models/classification/{model}/metrics.json') as f: 
            metrics = json.load(f)
            validation_acuracy = metrics['validation_accuracy']
            validation_recall = metrics['validation_recall']
            validation_precision = metrics['validation_precision']
            validation_f1_score = metrics['validation_f1_score']
            print(f'{model}: F1 score: {validation_f1_score}')

            if validation_f1_score < best_f1_score:
                best_f1_score = validation_f1_score
                best_model = model

    return print(f'The model with the lowest F1 Score is: {best_model}')


if __name__ == "__main__":
    X, y = load_airbnb(pd.read_csv('./airbnb-property-listings/tabular_data/clean_tabular_data.csv'), "Category")
    X_train, y_train, X_test, y_test, X_validation, y_validation = split_data(X, y)
    evaluate_all_models(X_train, y_train, X_test, y_test, X_validation, y_validation)
    find_best_model()