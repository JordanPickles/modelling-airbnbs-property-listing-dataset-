
import pandas as pd
import numpy as np
from numpy import array
import torch 
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder ,OneHotEncoder

import itertools
from itertools import product

from tabular_data import load_airbnb

import yaml
import os
import shutil
import json
import time

from datetime import datetime
from math import sqrt


class AirbnbBedroomDataset(Dataset):
    def __init__(self, data, prediction_variable):
        super().__init__() 

        numerical_columns = pd.DataFrame(data.select_dtypes(include=['int64', 'float64']))

        # Label encodes the category column in preparation for 
        le = LabelEncoder()
        data['Category'] = le.fit_transform(data['Category'])

        # fit and transform the category column
        ohe = OneHotEncoder(handle_unknown='ignore')
        category_encoded = pd.DataFrame(ohe.fit_transform(data['Category'].values.reshape(-1,1)).toarray())

        self.X = numerical_columns.join(category_encoded) 
        self.X = self.X.drop(df.columns[df.columns.str.contains('unnamed', case = False)], axis = 1)
        self.X = self.X.drop(columns = [prediction_variable], axis = 1)
        self.X = self.X.values

        # Encodes the labels
        self.y = le.fit_transform(data[f'{prediction_variable}'].values)

        assert len(self.X) == len(self.y) # Data and labels have to be of equal length

    def __getitem__(self, index):
        return (torch.tensor(self.X[index]), torch.tensor(self.y[index]))

    def __len__(self):
        return len(self.X)



class NN(torch.nn.Module):
    def __init__(self, nn_config):
        super().__init__()
        #define layers
        self.hidden_layer_width = nn_config['hidden_layer_width']
        self.dropout = nn_config['dropout']
        
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(16, self.hidden_layer_width), # uses the same width in all layers of the model
            torch.nn.ReLU(),       
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.hidden_layer_width, self.hidden_layer_width),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.hidden_layer_width, 9) #Number of unique bedrooms in the labels

        )

        
    def forward(self, X):
        return F.softmax(self.layers(X))

def split_data(dataset): #TODO scalar the data
    """Splits the input dataset into training, validation, and testing sets.

    Parameters:
        - dataset: a PyTorch Dataset object that contains the input data.

    Returns:
        - train_dataset: a PyTorch Dataset object containing 70% of the input data, to be used for training.
        - validation_dataset: a PyTorch Dataset object containing 15% of the input data, to be used for validation.
        - test_dataset: a PyTorch Dataset object containing 15% of the input data, to be used for testing."""

    # Splits data into 70% training and 30% test
    train_dataset, test_dataset = random_split(dataset, [int(len(dataset) * 0.7), len(dataset)-int(len(dataset)*0.7)])

    # Splits test data in half, percentage of total dataset is 15% test and 15% validation
    validation_dataset, test_dataset = random_split(test_dataset, [int(len(test_dataset) * 0.5), len(test_dataset)-int(len(test_dataset)*0.5)])

    print(f"    Training: {len(train_dataset)}")
    print(f"    Validation: {len(validation_dataset)}")
    print(f"    Testing: {len(test_dataset)}")


    return train_dataset, validation_dataset, test_dataset


def train_model(train_loader, validation_loader, nn_config, epochs=10):
    """Trains a neural network model on the provided training data and evaluates its performance on the validation data.

    Parameters:
        - train_loader (DataLoader): A PyTorch DataLoader object that loads the training data in batches.
        - validation_loader (DataLoader): A PyTorch DataLoader object that loads the validation data in batches.
        - nn_config (dict): A dictionary containing the configuration for the neural network, including the number of input and output features, the number of hidden layers and their sizes, the activation function to be used, and the type of optimizer and learning rate.
        - epochs (int): An integer indicating the number of times to loop through the entire dataset during training (default: 10).

    Returns:
        - dict: A dictionary containing the trained model, training and validation losses, and performance metrics such as accuracy, precision, recall, and F1-score.

    """

    
    model = NN(nn_config)
    writer = SummaryWriter()
    if nn_config['optimiser'] == 'SGD':
        optimiser = torch.optim.SGD
        optimiser = optimiser(model.parameters(), nn_config['learning_rate'])
    if nn_config['optimiser'] == 'Adam':
        optimiser = torch.optim.Adam
        optimiser = optimiser(model.parameters(), nn_config['learning_rate'] )
    batch_index = 0
    train_cross_entropy_loss = 0.0
    train_running_accuracy = 0.0
    train_running_precision = 0.0
    train_running_recall = 0.0
    train_running_f1_error = 0.0

    validation_cross_entropy_loss = 0.0
    validation_running_accuracy = 0.0
    validation_running_precision = 0.0
    validation_running_recall = 0.0
    validation_running_f1_error = 0.0


    for epoch in range(epochs): # Loops through the dataset a number of times

        for batch in train_loader: # Samples different batches of the data from the data loader
            
            X_train, y_train = batch # Sets features and labels from the batch
            X_train = X_train.type(torch.float32)
            y_train = y_train.type(torch.float32)

            train_prediction = model(X_train) 

            cross_entropy_loss = F.cross_entropy(train_prediction, y_train.long()) 
            cross_entropy_loss = cross_entropy_loss.type(torch.float32)
            cross_entropy_loss.backward() # Populates the gradients from the parameters of the smodel
            
            optimiser.step() #Optimisation step
            optimiser.zero_grad() # Resets the grad to zero as the grads are no longer useful as they were calculated when the model had different parameters
            
            writer.add_scalar('training_loss', cross_entropy_loss.item(), batch_index)
            batch_index += 1

            _, train_prediction = torch.max(train_prediction, 1)

            y_train_detached = y_train.detach().numpy()

            # Accuracy score the same as F1 error in this multiclass classification - precision and recall calculated using weighted average as micro would return the same score as the accuracy
            train_running_accuracy += accuracy_score(y_train_detached, train_prediction)
            train_running_precision += precision_score(y_train_detached, train_prediction, average="weighted")
            train_running_recall += recall_score(y_train_detached, train_prediction, average="weighted")

        
        for batch in validation_loader: # Samples different batches of the data from the data loader
            
            X_validation, y_validation = batch # Sets features and labels from the batch
            X_validation = X_validation.type(torch.float32)
            y_validation = y_validation.type(torch.float32)
                        
            validation_prediction = model(X_validation) 
            cross_entropy_loss = F.cross_entropy(validation_prediction, y_validation.long()) 
            cross_entropy_loss = cross_entropy_loss.type(torch.float32)
            cross_entropy_loss.backward() # Populates the gradients from the parameters of the smodel
            
            optimiser.step() #Optimisation step
            optimiser.zero_grad() # Resets the grad to zero as the grads are no longer useful as they were calculated when the model had different parameters

            writer.add_scalar('validation_loss', cross_entropy_loss.item(), batch_index)
            batch_index += 1

            #Finds the maximum value of the second torch tensor which provides the value with the highest probability
            _, validation_prediction = torch.max(validation_prediction, 1)
            y_validation_detached = y_validation.detach().numpy()

            # Accuracy score the same as F1 error in this multiclass classification - precision and recall calculated using weighted average as micro would return the same score as the accuracy
            validation_running_accuracy += accuracy_score(y_validation_detached, validation_prediction)
            validation_running_precision += precision_score(y_validation_detached, validation_prediction, average="weighted")
            validation_running_recall += recall_score(y_validation_detached, validation_prediction, average="weighted")


    # Normalises performance metrics to the number of samples passed through the model
    train_accuracy = train_running_accuracy / (epochs*len(train_loader))
    train_precision = train_running_precision / (epochs*len(train_loader))
    train_recall = train_running_recall / (epochs*len(train_loader))

    validation_accuracy = validation_running_accuracy / (epochs*len(validation_loader))
    validation_precision = validation_running_precision / (epochs*len(validation_loader))
    validation_recall = validation_running_recall / (epochs*len(validation_loader))


    model_name = datetime.fromtimestamp(datetime.timestamp(datetime.now())).strftime("%d-%m-%Y, %H_%M_%S")

    if not os.path.exists('./models/classification/neural_networks/trained_models'):
        os.makedirs('./models/classification/neural_networks/trained_models')
    model_path = f'./models/classification/neural_networks/trained_models/{model_name}'
    torch.save(model.state_dict(), model_path)

    return train_accuracy, train_precision, train_recall, validation_accuracy, \
        validation_precision, validation_recall, model_name


def test_model(nn_config, state_dict_path, dataloader):
    """
    Test a neural network model using the specified configuration, state dictionary path, and data loader.

    Parameters:
        - nn_config (dict): A dictionary containing the configuration parameters for the neural network model.
        - state_dict_path (str): The path to the state dictionary file containing the trained model's parameters.
        - dataloader (torch.utils.data.DataLoader): A PyTorch DataLoader object containing the test data.

    Returns:
       -  Tuple: A tuple containing the test results, including the cross-entropy loss, accuracy, precision, recall, F1 score, and inference latency.

    """
    model = NN(nn_config)
    writer = SummaryWriter()

    batch_index = 0
    test_cross_entropy_loss = 0.0
    test_running_accuracy = 0.0
    test_running_precision = 0.0
    test_running_recall = 0.0
    test_running_f1_error = 0.0

    
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)

    for batch in dataloader: # Samples different batches of the data from the data loader
        
        X_test, y_test = batch # Sets features and labels from the batch
        X_test = X_test.type(torch.float32)
        y_test = y_test.type(torch.float32)

        prediction_start_time = time.time()
        test_prediction = model(X_test) 
        inference_latency = (time.time() - prediction_start_time) / len(test_prediction)    
     
        #Calculates  Cross Entropy Loss
        cross_entropy_loss = F.cross_entropy(test_prediction, y_test.long()) 
        cross_entropy_loss = cross_entropy_loss.type(torch.float32)
        cross_entropy_loss.backward() # Populates the gradients from the parameters of the smodel

        writer.add_scalar('test_loss', cross_entropy_loss.item(), batch_index)
        batch_index += 1

        #Finds the maximum value of the second torch tensor which provides the value with the highest probability
        _, test_prediction = torch.max(test_prediction, 1)
        
        y_test_detached = y_test.detach().numpy()

        # Accuracy score the same as F1 error in this multiclass classification - precision and recall calculated using weighted average as micro would return the same score as the accuracy
        test_running_accuracy += accuracy_score(y_test_detached, test_prediction)
        test_running_precision += precision_score(y_test_detached, test_prediction, average="weighted")
        test_running_recall += recall_score(y_test_detached, test_prediction, average="weighted")

    test_accuracy = test_running_accuracy / len(dataloader)
    test_precision = test_running_precision / len(dataloader)
    test_recall = test_running_recall / len(dataloader)


    return test_accuracy, test_precision, test_recall,inference_latency
           

def get_nn_config(config_file = 'nn_config.yaml') -> dict:
    """Loads neural network configuration from a YAML file and returns as a dictionary
    
    Parameters:
        config_file: path to the .yaml file containing the hyperparameters
        
    Outputs:
        nn_config: dict containing the hyperparameters for the model"""
    
    with open(config_file, 'r') as f:
        nn_config = yaml.safe_load(f)
    return nn_config

def generate_nn_configs():
    """Generates a list of dictionaries, where each dictionary represents a set of hyperparameters to be used in a neural network model.

    Parameters: None

    Returns: A list of dictionaries. Each dictionary contains the following keys:
        - 'optimiser': a string indicating the optimiser to be used in the neural network (either 'SGD' or 'Adam').
        - 'learning_rate': a float indicating the learning rate to be used in the neural network.
        - 'hidden_layer_width': an integer indicating the number of neurons in the hidden layers of the neural network.
        - 'dropiut': an float indicating the ratio of nodes to miss out in the neural network.
        The list contains all possible combinations of these hyperparameters."""
    
    hyperparameters = {
        'optimiser': ['SGD', 'Adam'],
        'learning_rate': [0.0001, 0.001],
        'hidden_layer_width': [10, 12, 14],
        'dropout': [0.2, 0.3] 
    }
    keys = hyperparameters.keys()
    values = hyperparameters.values()
    hyperparameter_combinations = [dict(zip(keys, combination)) for combination in product(*values)]
    
    return hyperparameter_combinations

def find_best_nn(hyperparameters, train_dataset, validation_dataset, test_dataset):
    """Trains and tests neural networks on different hyperparameters, returning the best performing model.
    
    Inputs:
        - hyperparameters: a dictionary with hyperparameters to train neural networks on
        - train_dataset: PyTorch Dataset object of training data
        - validation_dataset: PyTorch Dataset object of validation data
        - test_dataset: PyTorch Dataset object of test data
    
    Returns:
        A tuple containing:
            - best_model_name: the name of the best performing trained model
            - best_model_path: the path to the best performing trained model
            - best_hyperparameters: the hyperparameters of the best performing trained model
            - performance_metrics: a dictionary of the best model's performance metrics on the train, validation, and test datasets
    """

    nn_configs_dict = generate_nn_configs()
    
    # Creates data loaders for the training, validation and test sets
    train_loader = DataLoader(train_dataset, batch_size = 16, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size = 16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size = 16, shuffle=True)

    best_accuracy = 0.0
    best_model_name = None
    best_model_path = None
    best_hyperparameters = None
    performance_metrics = {}

    for nn_config in nn_configs_dict:
        train_start_time = time.time()
        train_accuracy, train_precision, train_recall, validation_accuracy, validation_precision, validation_recall, model_name = train_model(train_loader, validation_loader, nn_config) 
        model_training_duration = time.time() - train_start_time
      
        # Logic applied to decide if model trained is better than the previous best model, if so the best parameters are updated
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_model_name = model_name
            best_model_path = f'./models/classification/neural_networks/trained_models/{best_model_name}'
            best_hyperparameters = nn_config
            test_accuracy, test_precision, test_recall, inference_latency = test_model(best_hyperparameters, best_model_path, test_loader) 
            #Adds perfromance metrics to dict
            performance_metrics['train_accuracy'] = train_accuracy
            performance_metrics['train_precision'] = train_precision
            performance_metrics['train_recall'] = train_recall

            performance_metrics['validation_accuracy'] = validation_accuracy  
            performance_metrics['validation_precision'] = validation_precision
            performance_metrics['validation_recall'] = validation_precision
 
            performance_metrics['test_accuracy'] = test_accuracy
            performance_metrics['test_precision'] = test_precision
            performance_metrics['test_recall'] = test_recall

            performance_metrics['training_duration'] = model_training_duration
            performance_metrics['inference_latency'] = inference_latency
    
            
    return best_model_name, best_model_path, best_hyperparameters, performance_metrics

def save_model(model_name, model_path, hyperparameters, metrics, prediction_variable):
    """Save the trained neural network model with the specified model name, hyperparameters and performance metrics for the prediction variable.

Parameters:
    - model_name (str): Name of the best trained model.
    - model_path (str): Path of the saved model state dictionary.
    - hyperparameters (dict): Hyperparameters used to train the model.
    - metrics (dict): Performance metrics of the trained model.
    - prediction_variable (str): The name of the variable that the model is predicting.

Returns:
    None """
    # Creates a folder for the best model trained and then moves model state dict into a folder specifically for that model   
    new_model_folder = os.makedirs(f'./models/classification/neural_networks/{prediction_variable}/{model_name}')
    new_model_path = f'./models/classification/neural_networks/{prediction_variable}/{model_name}/model.pt'
    shutil.move(model_path, new_model_path)


    #Saves hyperparameters and model performance metrics
    with open(f'./models/classification/neural_networks/{prediction_variable}/{model_name}/hyperparameters.json', 'w') as f:
        json.dump(hyperparameters, f)
    with open(f'./models/classification/neural_networks/{prediction_variable}/{model_name}/metrics.json', 'w') as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    prediction_variable = 'bedrooms'
    df = pd.read_csv('./airbnb-property-listings/tabular_data/clean_tabular_data.csv')
    dataset = AirbnbBedroomDataset(df, prediction_variable) #Creates an instance of the class
    train_dataset, validation_dataset, test_dataset = split_data(dataset)
    best_model_name, best_model_path, best_hyperparameters, performance_metrics  = find_best_nn(generate_nn_configs(),train_dataset, validation_dataset, test_dataset)
    save_model(best_model_name, best_model_path, best_hyperparameters, performance_metrics, prediction_variable)




    



    


   


