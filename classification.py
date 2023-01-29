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
    
    grid_search = GridSearchCV(model_class, hyperparameters, scoring = 'neg_mean_squared_error', cv = 5) 
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_hyperparameters = grid_search.best_params_

    # Provides Validation Metrics
    y_val_pred = best_model.predict(X_validation)
    y_val_accuracy = best.model(X_validation)


    # Provides test metrics
    y_test_pred = best_model.predict(X_test)


    
    # Map metrics to the performance metrics 
    performance_metrics

    
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

if __name__ == "__main__":
    X, y = load_airbnb(pd.read_csv('./airbnb-property-listings/tabular_data/clean_tabular_data.csv'), "Category")
    X_train, y_train, X_test, y_test, X_validation, y_validation = split_data(X, y)  
    classification_model(X_train, y_train, X_test, y_test, X_validation, y_validation)