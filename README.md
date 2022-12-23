# modelling-airbnbs-property-listing-dataset-
This porject is the 4th project of the AiCore data career accelerator programme and this project contributes to the data science aspect of the course.

The objective of this project was to build a framework to systematically train, tune, and evaluate models on several tasks that are tackled by the Airbnb team.

## Milestone 3
During this milestone, two python files were produced, tabular_data.py and prepare_image_data.py to clean and prepare both the tabular data and image data in this project. The pandas package was used in both files. The os context manager package and the PIL package from pillow was used to prepare the image data.

### Cleaning the tabular data
```
def remove_rows_with_missing_ratings(df):
    """This method drops all rows with null values included from the rating columns and drops any unnamed columns
    Output: df with rows with null values in rating columns removed and unnamed column removed"""
    df = df.dropna(subset = ["Cleanliness_rating", "Accuracy_rating", "Communication_rating", "Location_rating", "Check-in_rating", "Value_rating"])
    df= df.drop(df.columns[df.columns.str.contains('unnamed', case = False)], axis = 1)
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
    """This function replaces all missing data in the beds, bathrooms, bedrooms and guests columns with the value 1
    Output: df with null columns containing the number 1"""
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
    Output:Tuple of features and values of fields with non-text data"""
    df = df.drop(labels = ['ID', 'Category', 'Title', 'Description', 'Amenities', 'Location', 'url'], axis = 1)
    features = df.drop(label, axis = 1)
    labels = df[label]

    return features, labels

if __name__ == "__main__":
    df = pd.read_csv('./airbnb-property-listings/tabular_data/listing.csv')
    df = clean_tabular_data(df)
    df.to_csv('./airbnb-property-listings/tabular_data/clean_tabular_data.csv')
    label = 'Price_Night'
    features, labels = load_airbnb(df, label)
```
### Preparing the image data
```
def resize_images(base_dir, processed_images_dir):
    """This function opens the image, checks all images are in RGB format and resizes all images in to the same heigh as the smallest images heigh whilst maintaing the aspect ratio of the image, before saving the image in a processed_images folder.
        base_dir: requires the base directory where all the folders containing the image is stored on the local machine
        processed_images_dir: provides the directory of where the processed photos will be saved"""

    image_file_path_list = []
    smallest_height = float('inf')
    
    for sub_dir in os.listdir(base_dir):
        sub_dir_file_path = os.path.join(base_dir, sub_dir)
        if os.path.isdir(sub_dir_file_path):
            for file in os.listdir(sub_dir_file_path):
                if file.endswith('.jpg') or file.endswith('.png'):
                    with Image.open(os.path.join(sub_dir_file_path, file)) as img:
                        if img.mode == 'RGB':
                            image_file_path_list.append(os.path.join(sub_dir_file_path, file))
                            width, height = img.size
                            if height < smallest_height:
                                smallest_height = height
    print(smallest_height)
                        

    for file in image_file_path_list:
        with Image.open(file) as img:
            width, height = img.size
            aspect_ratio = width / height
            new_width = int(aspect_ratio * smallest_height) 
            resized_img = img.resize((new_width, smallest_height))
            resized_img.save(os.path.join(processed_images_dir, os.path.basename(file)))

if __name__ == '__main__':
    if not os.path.exists('./airbnb-property-listings/processed_images'):
        os.mkdir('./airbnb-property-listings/processed_images')

    base_dir = './airbnb-property-listings/images/'
    processed_images_dir = './airbnb-property-listings/processed_images/'
    resize_images(base_dir, processed_images_dir)
```


    
