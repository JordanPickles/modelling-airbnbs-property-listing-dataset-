
import pandas as pd


def remove_rows_with_missing_ratings(df):
    #TODO: removes the rows with missing values in these columns. It should take in the dataset as a pandas dataframe and return the same type.
    return df

def combine_description_strings(df):
    #TODO: The "Description" column contains lists of strings. You'll need to define a function called combine_description_strings which combines 
    # the list items into the same string. Unfortunately, pandas doesn't recognise the values as lists, but as strings whose contents are valid Python lists. 
    # You should look up how to do this (don't implement a from-scratch solution to parse the string into a list). 
    # The lists contain many empty quotes which should be removed. If you don't remove them before joining the list elements with a whitespace, they might
    # cause the result to contain multiple whitespaces in places. The function should take in the dataset as a pandas dataframe and return the same type. 
    # It should remove any records with a missing description, and also remove the "About this space" prefix which every description starts with.
    return df

def set_default_feature_values(df):
    #TODO: The "guests", "beds", "bathrooms", and "bedrooms" columns have empty values for some rows. Don't remove them, instead, define a function called 
    # set_default_feature_values, and replace these entries with the number 1. It should take in the dataset as a pandas dataframe and return the same type.
    return df

def clean_tabular_data(df):
    #TODO: Put all of the code that does this processing into a function called clean_tabular_data which takes in the raw dataframe, calls these functions
    # sequentially on the output of the previous one, and returns the processed data.
    remove_rows_with_missing_ratings(df)
    combine_description_strings(df)
    set_default_feature_values(df)


if __name__ == "__main__":
df = pd.read_csv('./tabular_data/AirBnbData.csv')
#clean_tabular_data()
# Save the processed data as clean_tabular_data.csv
