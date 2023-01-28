import pandas as pd

def remove_rows_with_missing_ratings(df):
    """This function drops all rows with null values included from the rating columns and drops any unnamed columns
    Output: df with rows with null values in rating columns removed and unnamed column removed"""
    df = df.dropna(subset = ["Cleanliness_rating", "Accuracy_rating", "Communication_rating", "Location_rating", "Check-in_rating", "Value_rating"])
    df= df.drop(df.columns[df.columns.str.contains('unnamed', case = False)], axis = 1)
    return df

def combine_description_strings(df):
    """This function  completes the following actions to the description column of the data: Removes rows with null values, removes 'About this space' from all rows, strips white spaces, parses the string into a list, removes all empty elements of the list and finally joins the list to be one string
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
    """This function ensures all values are the correct data type being as numerical values in the selcted columns and then replaces all missing data in the beds, bathrooms, bedrooms and guests columns with the value 1
    Output: df with null columns containing the number 1"""

    df["bedrooms"] = pd.to_numeric(df["bedrooms"], errors = 'coerce')
    df["guests"] = pd.to_numeric(df["guests"], errors = 'coerce')
    df["bathrooms"] = pd.to_numeric(df["bathrooms"], errors = 'coerce')
    df["beds"] = pd.to_numeric(df["beds"], errors = 'coerce')

    df[["beds", "bathrooms", "bedrooms", "guests"]] = df[["beds", "bathrooms", "bedrooms", "guests"]].fillna(1)

    return df

def clean_tabular_data(df):
    """This function calls the methods to clean the data
    Output: cleaned data frame"""
    df = remove_rows_with_missing_ratings(df)
    df = combine_description_strings(df)
    df = set_default_feature_values(df)
    return df
   
def load_airbnb(df, label: str) -> tuple:
    """This function prepares the data to be used in a ML model returning a features and values tuple
    Output:Tuple of features and labels of fields with non-text data"""
    labels = df[label]
    features = df.drop(['ID', 'Category', 'Title', 'Description', 'Amenities', 'Location', 'url'], axis = 1)
    if label in df.columns:
        features = df.drop(label, axis = 1)
    

    return features, labels

if __name__ == "__main__":
    df = pd.read_csv('./airbnb-property-listings/tabular_data/listing.csv')
    df = clean_tabular_data(df)
    df.to_csv('./airbnb-property-listings/tabular_data/clean_tabular_data.csv')
    label = 'Price_Night'
    features, labels = load_airbnb(df, label)
    


