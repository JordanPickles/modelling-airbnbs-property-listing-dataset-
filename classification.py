import pandas as pd
import numpy as np
from tabular_data import load_airbnb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn.preprocessing import LabelBinarizer

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import os
import joblib
import json

def split_data(X, y):
    """This function splits the data into a train, test and validate samples at a rate of 70%, 15% and 15% resepctively.
    Input: 
        tuples contatining features and labels in numerical form
    Output: 
        3 datasets containing tuples of features and labels from the original dataframe"""

    binariser = LabelBinarizer()  
    #Encodes the labels
    y_binarised = binariser.fit_transform(y.values.reshape(-1,1))
    # Converts array into a 1D array, notes the index of the highest number of array (positive classification) to transform the array to be 1D and passible into a model
    y_binarised = np.argmax(y_binarised, axis=1)
    #Splits the data into train and test data at 70% and 30% respectively
    X_train, X_test, y_train, y_test = train_test_split(X, y_binarised, test_size = 0.3)
    #Splits the test data into test and validation set at 15% each
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size = 0.5)

    return X_train, y_train, X_test, y_test, X_validation, y_validation


def classification_model(X_train, y_train, X_test, y_test, X_validation, y_validation):
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
    performance_metrics = {}
    
    grid_search = GridSearchCV(model_class, hyperparameters, scoring = 'f1_score', cv = 5) 
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_hyperparameters = grid_search.best_params_

    # Provides Validation Metrics
    y_validation_pred = best_model.predict(X_validation)
    y_validation_accuracy = accuracy_score(y_validation, y_validation_pred)
    y_validation_precision = precision_score(y_validation, y_validation_pred)
    y_validation_recall = recall_score(y_validation, y_validation_pred)
    y_validation_f1 = f1_score(y_validation, y_validation_pred)

    # Provides test metrics
    y_test_pred = best_model.predict(X_test)    
    y_test_accuracy = accuracy_score(y_test, y_test_pred)
    y_test_precision = precision_score(y_test, y_test_pred)
    y_test_recall = recall_score(y_test, y_test_pred)
    y_test_f1 = f1_score(y_test, y_test_pred)

    # Maps metrics to the performance metrics dict
    performance_metrics['Validation Accuracy'] = y_validation_accuracy
    performance_metrics['Validation Precision'] = y_validation_precision
    performance_metrics['Validation Recall'] = y_validation_recall
    performance_metrics['Validation F1 Score'] = y_validation_f1

    performance_metrics['Test Accuracy'] = y_test_accuracy
    performance_metrics['Test Precision'] = y_test_precision
    performance_metrics['Test Recall'] = y_test_recall
    performance_metrics['Test F1 Score'] = y_test_f1
    
    return best_model, best_hyperparameters, performance_metrics

def save_model(model, hyperparameters, metrics, model_folder):
    
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    joblib.dump(model, f'{model_folder}/model.joblib')
    with open(f'{model_folder}/hyperparameters.json', 'w') as f:
        json.dump(hyperparameters, f)
    with open(f'{model_folder}/metrics.json', 'w') as f:
        json.dump(metrics, f)


def evaluate_all_models(X_train, y_train, X_test, y_test, X_validation, y_validation): 
    #TODO add docstrings
    
    # TODO Add Hyperparameters for random forest, decision tree and gradient boosting classifier mechanisms
    logistic_regression = {}



    # TODO Add dict of models
    classifier_models_dict = {'Logistic Regression': []}
    
    # Create required directories to save the models to   
    if not os.path.exists('./models'):
        os.makedirs('./models')
    if not os.path.exists('./models/regression'):
        os.makedirs('./models/regression')
    if not os.path.exists('./models/regression/logistic_regression'):
        os.makedirs('./models/regression/logistic_regression')

    # For loop iterates through the models provided and calls the tune_regression_mode_hyperparameters
    for key, values in models_dict.items(): #TODO - should a random seed be included here?
        model, hyperparameters = values
        best_model, best_hyperparameters, performance_metrics = tune_classification_model_hyperparameters(model, X_train, y_train, X_test, y_test, X_validation, y_validation, hyperparameters)
        folder_path = f'./models/regression/logistic_regression/{key}'
        save_model(best_model, best_hyperparameters, performance_metrics, folder_path) 
        

def find_best_model(): # TODO needs to take in a keyword argument called task_folder
    """This function compares the Root Mean Squared Error (RMSE) of the trained models on validation set and returns the model with the lowest RMSE.
    Inputs:
        None
    Outputs:
        Prints the model name with the lowest RMSE
    """
    #TODO change the models
    models = ['Logistic Regression', 'Decision Tree Classifier', 'Random Forest Classifier', 'Gradient Boosting Classifier']
    best_model = None
    best_rmse = float('inf') # Change the metric to accuracy or f1
    
    for model in models:
        with open(f'./models/regression/logistic_regression/{model}/metrics.json') as f: 
            metrics = json.load(f)

            # TODO Change metrics to classifcation relevant
            validation_r2 = metrics['validation_r2']
            validation_rmse = metrics['validation_rmse']
            validation_mae = metrics['validation_mae']
            print(f'{model}: RMSE: {validation_rmse}')

            # TODO Adapt to correct metrics
            if validation_rmse < best_rmse:
                best_rsme = validation_rmse
                best_model = model

if __name__ == "__main__":
    X, y = load_airbnb(pd.read_csv('./airbnb-property-listings/tabular_data/clean_tabular_data.csv'), "Category")
    X_train, y_train, X_test, y_test, X_validation, y_validation = split_data(X, y)  
    classification_model(X_train, y_train, X_test, y_test, X_validation, y_validation)