import pandas as pd
import numpy as np
from tabular_data import load_airbnb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


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
    
    grid_search = GridSearchCV(model_class, hyperparameters, scoring = 'accuracy', cv = 5) 
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


    # Map metrics to the performance metrics 
    performance_metrics['Validation Accuracy'] = y_validation_accuracy
    performance_metrics['Validation Precision'] = y_validation_precision
    performance_metrics['Validation Recall'] = y_validation_recall
    performance_metrics['Validation F1 Score'] = y_validation_f1

    performance_metrics['Test Accuracy'] = y_test_accuracy
    performance_metrics['Test Precision'] = y_test_precision
    performance_metrics['Test Recall'] = y_test_recall
    performance_metrics['Test F1 Score'] = y_test_f1

    
    return best_model, best_hyperparameters, performance_metrics



if __name__ == "__main__":
    X, y = load_airbnb(pd.read_csv('./airbnb-property-listings/tabular_data/clean_tabular_data.csv'), "Category")
    X_train, y_train, X_test, y_test, X_validation, y_validation = split_data(X, y)  
    classification_model(X_train, y_train, X_test, y_test, X_validation, y_validation)