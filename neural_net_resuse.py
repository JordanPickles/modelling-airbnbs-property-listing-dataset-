
import pandas as pd
import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

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
        self.data = data.drop(data.columns[data.columns.str.contains('unnamed', case = False)], axis = 1)
        self.X, self.y = load_airbnb(self.data, prediction_variable)
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
            torch.nn.Linear(11, self.hidden_layer_width), # uses the same width in all layers of the model
            torch.nn.ReLU(),       
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.hidden_layer_width, self.hidden_layer_width),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.hidden_layer_width, 1)

        )

        
    def forward(self, X):
        return F.sigmoid(self.layers(X))

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
        - train_loader: a PyTorch DataLoader object that loads the training data in batches.
        - validation_loader: a PyTorch DataLoader object that loads the validation data in batches.
        - nn_config: a dictionary containing the configuration for the neural network, including the number of input and output features, the number of hidden layers and their sizes, the activation function to be used, and the type of optimizer and learning rate.
        - epochs: an integer indicating the number of times to loop through the entire dataset during training (default: 10).

    Returns:
        - train_rmse_loss: a float representing the root mean squared error (RMSE) of the model's predictions on the training data.
        - validation_rmse_loss: a float representing the RMSE of the model's predictions on the validation data.
        - train_r2: a float representing the R-squared score of the model's predictions on the training data.
        - validation_r2: a float representing the R-squared score of the model's predictions on the validation data.
        - model_name: a string representing the name of the trained model file."""
    
    model = NN(nn_config)
    writer = SummaryWriter()
    scaler = MinMaxScaler()
    if nn_config['optimiser'] == 'SGD':
        optimiser = torch.optim.SGD
        optimiser = optimiser(model.parameters(), nn_config['learning_rate'])
    if nn_config['optimiser'] == 'Adam':
        optimiser = torch.optim.Adam
        optimiser = optimiser(model.parameters(), nn_config['learning_rate'] )
    batch_index = 0
    train_cross_entropy_loss = 0.0
    validation_cross_entropy_loss = 0.0


    for epoch in range(epochs): # Loops through the dataset a number of times

        for batch in train_loader: # Samples different batches of the data from the data loader
            
            X_train, y_train = batch # Sets features and labels from the batch

            X_train = X_train.type(torch.float32)
            y_train = y_train.type(torch.float32)
            y_train = y_train.view(-1, 1)

                        
            train_prediction = model(X_train) 
            cross_entropy_loss = F.binary_cross_entropy(train_prediction, y_train) 
            cross_entropy_loss = cross_entropy_loss.type(torch.float32)
            cross_entropy_loss.backward() # Populates the gradients from the parameters of the smodel
            
            optimiser.step() #Optimisation step
            optimiser.zero_grad() # Resets the grad to zero as the grads are no longer useful as they were calculated when the model had different parameters

            writer.add_scalar('training_loss', cross_entropy_loss.item(), batch_index)
            batch_index += 1

            # TODO Add other metrics? F1, recall, precision and accuracy?


        for batch in validation_loader: # Samples different batches of the data from the data loader
            
            X_validation, y_validation = batch # Sets features and labels from the batch
            X_validation = X_validation.type(torch.float32)
            # X_validation_scaled = torch.tensor(scaler.fit_transform(X_validation))
            y_validation = y_validation.type(torch.float32)
            y_validation = y_validation.view(-1, 1)
                        
            validation_prediction = model(X_validation) 
            cross_entropy_loss = F.binary_cross_entropy(validation_prediction, y_validation) 
            cross_entropy_loss = cross_entropy_loss.type(torch.float32)
            cross_entropy_loss.backward() # Populates the gradients from the parameters of the smodel
            
            optimiser.step() #Optimisation step
            optimiser.zero_grad() # Resets the grad to zero as the grads are no longer useful as they were calculated when the model had different parameters

            writer.add_scalar('training_loss', cross_entropy_loss.item(), batch_index)
            batch_index += 1


            # TODO Add other metrics? F1, recall, precision and accuracy?



    #Normalises performance metrics to the number of samples passed through the model
    train_cross_entropy_loss = train_cross_entropy_loss/(epochs*len(train_loader))
    validation_cross_entropy_loss = validation_cross_entropy_loss/(epochs*len(validation_loader))

    model_name = datetime.fromtimestamp(datetime.timestamp(datetime.now())).strftime("%d-%m-%Y, %H:%M:%S")

    if not os.path.exists('./models/classification/neural_networks/trained_models'):
        os.makedirs('./models/classification/neural_networks/trained_models')
    model_path = f'./models/classification/neural_networks/trained_models/{model_name}'
    torch.save(model.state_dict(), model_path)

    return train_cross_entropy_loss, validation_cross_entropy_loss, model_name


def test_model(nn_config, state_dict_path, dataloader):
    """
    This function tests the trained neural network model using the test dataset and calculates its RMSE loss and R2 score.

    Parameters:
        - nn_config: a dictionary containing the configuration of the neural network
        - state_dict_path: the file path of the state dictionary for the trained model
        - dataloader: a PyTorch dataloader object that loads the test data
    
    Returns:
        - test_rmse_loss: the root mean squared error loss of the model on the test data
        - inference_latency: the time taken to make a single prediction
        - test_r2: the R2 score of the model on the test data
    """
    
    model = NN(nn_config)
    writer = SummaryWriter()
    scaler = MinMaxScaler()

    batch_index = 0
    test_cross_entropy_loss = 0.0

    
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)

    for batch in dataloader: # Samples different batches of the data from the data loader
        
        X_test, y_test = batch # Sets features and labels from the batch
        X_test = X_test.type(torch.float32)
        # X_test_scaled = torch.tensor(scaler.fit_transform(X_test))
        y_test = y_test.type(torch.float32)
        y_test = y_test.view(-1, 1)
       

        prediction_start_time = time.time()
        test_prediction = model(X_test) 
        inference_latency = (time.time() - prediction_start_time) / len(test_prediction)    
     
        #Calculate Loss
        cross_entropy_loss = F.binary_cross_entropy(test_prediction, y_test) 
        cross_entropy_loss = cross_entropy_loss.type(torch.float32)
        cross_entropy_loss.backward() # Populates the gradients from the parameters of the smodel

        batch_index += 1
    
    test_cross_entropy_loss = test_cross_entropy_loss/(len(dataloader))

    return test_cross_entropy_loss, inference_latency
           

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
        - 'model_depth': an integer indicating the number of hidden layers in the neural network.
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
            - best_model_metrics: a dictionary of the best model's performance metrics on the train, validation, and test datasets
    """

    nn_configs_dict = generate_nn_configs()
    
    train_loader = DataLoader(train_dataset, batch_size = 16, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size = 16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size = 16, shuffle=True)

    best_cross_entropy_loss = np.inf
    best_model_name = None
    best_model_path = None
    best_hyperparameters = None
    performance_metrics = {}


    for nn_config in nn_configs_dict:
        
        train_start_time = time.time()
        train_cross_entropy_loss, validation_cross_entropy_loss, model_name = train_model(train_loader, validation_loader, nn_config) 
        model_training_duration = time.time() - train_start_time
      
        # Logic applied to decide if model trained is better than the previous best model, if so the best parameters are updated
        if validation_cross_entropy_loss < best_cross_entropy_loss:
            best_cross_entropy_loss = validation_cross_entropy_loss
            best_model_name = model_name
            best_model_path = f'./models/regression/neural_networks/trained_models/{best_model_name}'
            best_hyperparameters = nn_config
            best_model_metrics = performance_metrics
            test_cross_entropy_loss, inference_latency = test_model(best_hyperparameters, best_model_path, test_loader) 
            #Adds perfromance metrics to dict
            performance_metrics['train_cross_entropy_loss'] = train_cross_entropy_loss
            
            performance_metrics['validation_cross_entropy_loss'] = validation_cross_entropy_loss
            
            performance_metrics['test_cross_entropy_loss'] = test_cross_entropy_loss
            
            performance_metrics['training_duration'] = model_training_duration
            performance_metrics['inference_latency'] = inference_latency
    
            
    return best_model_name, best_model_path, best_hyperparameters, best_model_metrics


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




    



    


   


