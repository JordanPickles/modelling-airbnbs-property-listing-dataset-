import pandas as pd
import numpy as np
from tabular_data import load_airbnb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score


def split_data(X, y):
    """This function splits the data into a train, test and validate samples at a rate of 70%, 15% and 15% resepctively.
    Input: 
        tuples contatining features and labels in numerical form
    Output: 
        3 datasets containing tuples of features and labels from the original dataframe"""
    print(y.unique())
    encoder = OneHotEncoder()
    y_encoded = encoder.fit_transform(y.values.reshape(-1, 1))
    print(y_encoded)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size = 0.3)

    # print("Number of samples in:")
    # print(f"    Training: {len(y_train)}")
    # print(f"    Testing: {len(y_test)}")

    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size = 0.5)

    # print("Number of samples in:")
    # print(f"    Training: {len(y_train)}")
    # print(f"    Testing: {len(y_test)}")
    # print(f"    Validation: {len(y_validation)}")

    return X_train, y_train, X_test, y_test, X_validation, y_validation

def classification_model(X_train, y_train, X_test, y_test, X_validation, y_validation):
    


    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(accuracy_score(y_test, y_pred))






if __name__ == "__main__":
    X, y = load_airbnb(pd.read_csv('./airbnb-property-listings/tabular_data/clean_tabular_data.csv'), "Category")
    X_train, y_train, X_test, y_test, X_validation, y_validation = split_data(X, y)  
    classification_model(X_train, y_train, X_test, y_test, X_validation, y_validation)