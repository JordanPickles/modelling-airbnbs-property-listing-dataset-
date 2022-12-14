import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tabular_data import load_airbnb

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

    train_loss = mean_squared_error(y_train, y_train_pred)
    validation_loss = mean_squared_error(y_validation, y_validation_pred)
    test_loss = mean_squared_error(y_test, y_test_pred)

    print(f"Train Loss: {train_loss} | Validation Loss: {validation_loss} | "f"Test Loss: {test_loss}")


if __name__ == "__main__":
    X, y = load_airbnb(pd.read_csv('./airbnb-property-listings/tabular_data/clean_tabular_data.csv'), 'Price_Night')
    X_train, y_train, X_test, y_test, X_validation, y_validation = split_data(X, y)
    train_model(X_train, y_train, X_test, y_test, X_validation, y_validation)