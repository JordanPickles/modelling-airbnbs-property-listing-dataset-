import pandas as pd

def remove_rows_with_missing_ratings(df):
    """This method drops all rows with null values included from the rating columns
    Output: df with rows with null values in rating columns removed"""
    df = df.dropna(subset = ["Cleanliness_rating", "Accuracy_rating", "Communication_rating", "Location_rating", "Check-in_rating", "Value_rating"])
    return df

def combine_description_strings(df):
    """This method completes the following actions to the description column of the data: Removes rows with null values, removes 'About this space' from all rows, strips white spaces, parses the string into a list, removes all empty elements of the list and finally joins the list to be one string
    Output: df with the description data cleaned """
    df.dropna(subset = ['Description'])
    df['Description'] = df['Description'].astype(str)
    df['Description'] = df['Description'].str.replace('About this space', '')
    df['Description'] = df['Description'].str.replace("'", '')
    df['Description'] = df['Description'].apply(lambda x: x.strip())
    df['Description'] = df['Description'].str.split(",")
    df['Description'] = df['Description'].apply(lambda x: [i for i in x if i != ''])
    df['Description'] = df['Description'].apply(lambda x: ''.join(x)).astype(str)

    return df

def set_default_feature_values(df):
    """This method replaces all missing data in the beds, bathrooms, bedrooms and guests columns with the value 1
    Output: df with null columns containing the number 1"""
    df[["beds", "bathrooms", "bedrooms", "guests"]] = df[["beds", "bathrooms", "bedrooms", "guests"]].fillna(1)

    return df

def clean_tabular_data(df):
    #TODO: Put all of the code that does this processing into a function called clean_tabular_data which takes in the raw dataframe, calls these functions
    # sequentially on the output of the previous one, and returns the processed data.
    """This method calls the methods to clean the data
    Output: cleaned data frame"""
    df = remove_rows_with_missing_ratings(df)
    df = combine_description_strings(df)
    df = set_default_feature_values(df)
   
    return df


if __name__ == "__main__":
    df = pd.read_csv('./airbnb-property-listings/tabular_data/listing.csv')
    df = clean_tabular_data(df)
    df.to_csv('./airbnb-property-listings/tabular_data/clean_tabular_data.csv')


