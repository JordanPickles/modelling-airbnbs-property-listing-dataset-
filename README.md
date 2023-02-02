# modelling-airbnbs-property-listing-dataset-
This porject is the 4th project of the AiCore data career accelerator programme and this project contributes to the data science aspect of the course.

The objective of this project was to build a framework to systematically train, tune, and evaluate models on several tasks that are tackled by the Airbnb team.

## Milestone 1
During this milestone, two python files were produced, tabular_data.py and prepare_image_data.py to clean and prepare both the tabular data and image data in this project. The pandas package was used in both files. The os context manager package and the PIL package from pillow was used to prepare the image data.

### Cleaning the tabular data
In this module, the data was taken in from a .csv file, cleaned and prepared for modelling in later milestones. Data with missing values were dropped form the df, text data was formatted and cleaned and the data. The data was returned as tuples (features and labels) to be used by a machine learning model.

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
This module takes in the images from the repository and formats all of the images by findings the smallest width of the images and resizes all images to the same height whilst maintaining the images aspect ratio. 
```
def resize_images(base_dir, processed_images_dir):
    """This function opens the image, checks all images are in RGB format and resizes all images in to the same height whilst maintaing the aspect ratio of the image, before saving the image in a processed_images folder.
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

## Milestone 2
During this milestone, a regression model was created using several regression algorithms to predict the price per night from several features. Initially the data was loaded by calling the load_airbnb() method from the tabular_data.py module previously created. The data was then split using train_test_split at a 70%, 15% and 15% ratio.

```
def split_data(X, y):
    """This function splits the data into a train, test and validate samples at a rate of 70%, 15% and 15% resepctively.
    Input: 
        tuples contatining features and labels in numerical form
    Output: 
        3 datasets containing tuples of features and labels from the original dataframe"""
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
```

The regression models were then created, the following models were deployed:
- Stochastic Gradient Descent Regressor - Aim to find the best set of model parameters by iteratively adjusting the weights of the parameters to reduce the gradient of the loss function

- Random Forest Regressor - Creates a random ensemble of decision trees, which are not influenced by the previous tree, to predict an outcome. To reduce the error the model aggregates the outputs of all of the trees.

- Decision Tree Regressor - Creates a decision tree model to predict the output by inputting rules to be followed down a tree

- Gradient Boosting Regressor - Creates an ensemble of decision trees which are influenced by the previous tree, each tree works iteratively to reduce the error of the previous tree.

Each model underwent hyperparameter tuning using gridsearch to find the opitmal hyperparameters of each model. For each model, evaluation metrics were assessed and saved in the repository to be assessed later.

```
def tune_regression_model_hyperparameters(model_class, X_train, y_train, X_test, y_test, X_validation, y_validation, hyperparameters):
    """
    This function takes in a regression model class, training, testing and validation datasets, and a dictionary of hyperparameters to tune. It then uses GridSearchCV to find the best hyperparameters for the model, and returns the best model, the best hyperparameters, and performance metrics (validation and test rmse, r2, and mae).

    Inputs:
        model_class: a regression model class (e.g. RandomForestRegressor)
        X_train: training set of features
        y_train: training set of labels
        X_test: testing set of features
        y_test: testing set of labels
        X_validation: validation set of features
        y_validation: validation set of labels
        hyperparameters: dictionary of hyperparameters to tune
    Outputs:
        best_model: an instance of the model_class, with the best hyperparameters found
        best_hyperparameters: a dictionary of the best hyperparameters found
        performance_metrics: a dictionary of performance metrics (validation and test rmse, r2, and mae)
    """

    performance_metrics = {}
    
    grid_search = GridSearchCV(model_class, hyperparameters, scoring = 'neg_mean_squared_error', cv = 5) 
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_hyperparameters = grid_search.best_params_

    # Provides Validation Metrics
    y_val_pred = best_model.predict(X_validation)
    best_validation_rmse = sqrt(-grid_search.best_score_)
    validation_r2 = r2_score(y_validation, y_val_pred)
    validation_mae = mean_absolute_error(y_validation, y_val_pred)

    # Provides test metrics
    y_test_pred = best_model.predict(X_test)
    test_rmse = sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    
    # Map metrics to the performance metrics 
    performance_metrics['validation_rmse'] = best_validation_rmse
    performance_metrics['validation_r2'] = validation_r2
    performance_metrics['validation_mae'] = validation_mae
    performance_metrics['test_rmse'] = test_rmse 
    performance_metrics['test_r2'] = test_r2
    performance_metrics['test_mae'] = test_mae
    
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
    
    def evaluate_all_models(X_train, y_train, X_test, y_test, X_validation, y_validation): 
    """This function evaluate different regression models by tuning the hyperparameters and then saving the best models, hyperparameters and performance metrics to specific folder.

    Inputs:
        X_train: A numpy array or pandas dataframe, representing the training set for the independent variables.
        y_train: A numpy array or pandas dataframe, representing the training set for the dependent variable.
        X_test: A numpy array or pandas dataframe, representing the testing set for the independent variables.
        y_test: A numpy array or pandas dataframe, representing the testing set for the dependent variable.
        X_validation: A numpy array or pandas dataframe, representing the validation set for the independent variables.
        y_validation: A numpy array or pandas dataframe, representing the validation set for the dependent variable.
    
    Outputs:
        It saves the best models, hyperparameters and performance metrics of all evaluated models to specific folder."""

    sgd_hyperparameters= {
        'penalty': ['l2', 'l1','elasticnet'],
        'alpha': [0.1, 0.01, 0.001, 0.0001],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9 ],
        'max_iter': [500, 1000, 1500, 2000],
        'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive']
    }

    decision_tree_hyperparameters = {
    'max_depth': [10, 20, 50],
    'min_samples_split': [2, 4, 6, 8],
    'min_samples_leaf': [1, 3, 5, 7],
    'splitter': ['best', 'random'] 
    }
    random_forest_hyperparameters = {
        'n_estimators': [50, 100, 150],
        'max_depth': [10,20,50],
        'min_samples_split': [2, 4, 6, 8],
        'min_samples_leaf': [1, 3, 5, 7]

    }
    gradient_boost_hyperparameters = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.1, 0.001, 0.0001],
        'criterion': ['friedman_mse', 'squared_error'],
        'min_samples_split': [2, 4, 6, 8],
        'min_samples_leaf': [1, 3, 5, 7]
    }
    model_hyperparameters = [decision_tree_hyperparameters, random_forest_hyperparameters, gradient_boost_hyperparameters]

    models_dict = {
        'SGD Regressor': [SGDRegressor(), sgd_hyperparameters],
        'Decision Tree Regressor': [DecisionTreeRegressor(), decision_tree_hyperparameters],
        'Random Forest Regressor': [RandomForestRegressor(), random_forest_hyperparameters],
        'Gradient Boosting Regressor': [GradientBoostingRegressor(), gradient_boost_hyperparameters]

    }
    
    # Create required directories to save the models to   
    if not os.path.exists('./models'):
        os.makedirs('./models')
    if not os.path.exists('./models/regression'):
        os.makedirs('./models/regression')
    if not os.path.exists('./models/regression/linear_regression'):
        os.makedirs('./models/regression/linear_regression')

    # For loop iterates through the models provided and calls the tune_regression_mode_hyperparameters
    for key, values in models_dict.items(): #TODO - should a random seed be included here?
        model, hyperparameters = values
        best_model, best_hyperparameters, performance_metrics = tune_regression_model_hyperparameters(model, X_train, y_train, X_test, y_test, X_validation, y_validation, hyperparameters)
        folder_path = f'./models/regression/linear_regression/{key}'
        save_model(best_model, best_hyperparameters, performance_metrics, folder_path) 

```

The evaluation metrics were then assessed to find the model with the greates accuracy.

```
def find_best_model():
    """This function compares the Root Mean Squared Error (RMSE) of the trained models on validation set and returns the model with the lowest RMSE.
    Inputs:
        None
    Outputs:
        Prints the model name with the lowest RMSE
    """
    models = ['SGD Regressor', 'Decision Tree Regressor', 'Random Forest Regressor', 'Gradient Boosting Regressor']
    best_model = None
    best_rmse = float('inf')
    best_r2 = 0
    for model in models:
        with open(f'./models/regression/linear_regression/{model}/metrics.json') as f: 
            metrics = json.load(f)
            validation_r2 = metrics['validation_r2']
            validation_rmse = metrics['validation_rmse']
            validation_mae = metrics['validation_mae']
            print(f'{model}: RMSE: {validation_rmse}')

            if validation_rmse < best_rmse:
                best_rsme = validation_rmse
                best_model = model

    
    return print('The model with the lowest RMSE is: {best_model}')
```

To call the methods an if__name__ == "__main__" section was created.

```
if __name__ == "__main__":
    X, y = load_airbnb(pd.read_csv('./airbnb-property-listings/tabular_data/clean_tabular_data.csv'), 'Price_Night')
    X_train, y_train, X_test, y_test, X_validation, y_validation = split_data(X, y) 
    evaluate_all_models(X_train, y_train, X_test, y_test, X_validation, y_validation)
    find_best_model()
```

## Milestone 3 

This model added a classifier model to make predictions on the property category by training classification models. The models trained and evaluated included; Logistic Regression, Decision Tree, Gradient Boosting Classifier and Random Forest Classifier.

To develop this module, several functions were adapted from the linear regression model, the primary changes are with the error measure of the model and the evaluation metrics used, f1_micro was the scoring measure for the grid search conducted.

```
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


```

Hyperparameters were set for the Logistic Regression, Decision Tree, Random Forest and Gradient Boosting classifiers. The hyperparameter tuning was conudcted using GridSearch and then the optimised model was trained and used to predict values of the validation set by calling the tune_classification_model_hyperparameters() function.

```
def evaluate_all_models(X_train, y_train, X_test, y_test, X_validation, y_validation): 
    
    
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
        'min_samples_leaf': [1, 2, 3, 4],
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
    for key, values in classification_models_dict.items(): #TODO - should a random seed be included here?
        model, hyperparameters = values
        best_model, best_hyperparameters, performance_metrics = tune_classification_model_hyperparameters(model, X_train, y_train, X_test, y_test, X_validation, y_validation, hyperparameters)
        folder_path = f'./models/classification/{key}'
        save_model(best_model, best_hyperparameters, performance_metrics, folder_path) 
```

To evaluate the performance of the models, F1 error score was used to evaluate the model performance.

```
def find_best_model(): 
    """This function compares the Root Mean Squared Error (RMSE) of the trained models on validation set and returns the model with the lowest RMSE.
    Parameters:
        None
    Outputs:
        Prints the model name with the lowest RMSE
    """
    
    models = ['Logistic Regression', 'Decision Tree Classifier', 'Random Forest Classifier', 'Gradient Boosting Classifier']
    best_model = None
    best_f1_score= float('inf') # Change the metric to accuracy or f1
    
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
```
