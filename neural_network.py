
import pandas as pd
import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

import itertools
from itertools import product

from tabular_data import load_airbnb
import yaml
import os
import json
import time
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt


class AirbnbNightlyPriceImageDataset(Dataset):
    def __init__(self):
        super().__init__() 
        self.data = pd.read_csv('./airbnb-property-listings/tabular_data/clean_tabular_data.csv')
        self.X, self.y = load_airbnb(self.data, 'Price_Night')
        assert len(self.X) == len(self.y) # Data and labels have to be of equal length
        print(self.X.shape)
        print(self.y.shape)

    def __getitem__(self, index):
        return (torch.tensor(self.X[index]), torch.tensor(self.y[index]))

    def __len__(self):
        return len(self.X)


class NN(torch.nn.Module):
    def __init__(self, nn_config):
        super().__init__()
        #define layers
        self.nn_config = nn_config
        self.hidden_layer_width = nn_config['hidden_layer_width']
        self.depth = nn_config['model_depth']
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_layer_width, self.hidden_layer_width), # uses the same width in all layers of the model
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_layer_width, 1) 
        )
    def forward(self, X):
        return self.layers(X)

def split_data(dataset): #TODO scalar the data
    # Splits data into 70% training and 30% test
    train_dataset, test_dataset = random_split(dataset, [int(len(dataset) * 0.7), len(dataset)-int(len(dataset)*0.7)])

    # Splits test data in half, percentage of total dataset is 15% test and 15% validation
    validation_dataset, test_dataset = random_split(test_dataset, [int(len(test_dataset) * 0.5), len(test_dataset)-int(len(test_dataset)*0.5)])

    print(f"    Training: {len(train_dataset)}")
    print(f"    Validation: {len(validation_dataset)}")
    print(f"    Testing: {len(test_dataset)}")


    return train_dataset, validation_dataset, test_dataset


# Function that trains the model
def train_model(model, train_loader, validation_loader, nn_config, epochs=10):
    writer = SummaryWriter()
    if nn_config['optimiser'] == 'SGD':
        optimiser = torch.optim.SGD
        optimiser = optimiser(model.parameters(),nn_config['learning_rate'])
    batch_index = 0

    min_validation_loss = np.inf

    train_start_time = time.time()
    for epoch in range(epochs): # Loops through the dataset a number of times
        
        train_loss = 0.0
        validation_loss = 0.0
        
        for batch in train_loader: # Samples different batches of the data from the data loader
            
            X, y = batch # Sets features and labels from the batch
            X = X.type(torch.float32)
            y = y.type(torch.float32)
                        
            train_prediction = model(X) 
            loss = F.mse_loss(train_prediction, y) 
            loss = loss.type(torch.float32)
            loss.backward() # Populates the gradients from the parameters of our model
            
            optimiser.step() #Optimisation step
            optimiser.zero_grad() # Resets the grad to zero as the grads are no longer useful as they were calculated when the model had different parameters

            writer.add_scalar('loss', loss.item(), batch_index)
            batch_index += 1

            train_loss += loss.item()


        for batch in validation_loader: # Samples different batches of the data from the data loader
            
            X, y = batch # Sets features and labels from the batch
            X = X.type(torch.float32)
            y = y.type(torch.float32)
                        
            validation_prediction = model(X) 
            #add different loss measures per requirement, e.g. rmse, r2 etc
            loss = F.mse_loss(validation_prediction, y) #TODO work out how to calculate this accurately
            loss = loss.type(torch.float32) 

            validation_loss += loss

    return train_loss, validation_loss


def test_model(model, dataloader, epochs=10):
    test_loss = 0.0
    for batch in dataloader: # Samples different batches of the data from the data loader
        
        X, y = batch # Sets features and labels from the batch
        X = X.type(torch.float32)
        y = y.type(torch.float32)
       
        prediction_start_time = time.time()
        prediction = model(X) 
        prediction_end_time = time.time()
        inference_latency = prediction_end_time - prediction_start_time / len() # TODO move the prediction times into the train or accuracy function?            
        
        #add different loss measures per requirement, e.g. rmse, r2 etc
        loss = F.mse_loss(prediction, y) #TODO work out how to calculate this accurately
        loss = loss.type(torch.float32) 
        test_loss += loss

    return test_loss, inference_latency
           

def get_nn_config(config_file = 'nn_config.yaml') -> dict:
    with open(config_file, 'r') as f:
        nn_config = yaml.safe_load(f)
    return nn_config

def generate_nn_configs():
    hyperparameters = {
        'optimiser': ['SGD', 'Adam'],
        'learning_rate': [0.001, 0.01, ],
        'hidden_layer_width': [10, 12],
        'model_depth': [3, 4] 
    }
    keys = hyperparameters.keys()
    values = hyperparameters.values()
    hyperparameter_combinations = [dict(zip(keys, combination)) for combination in product(*values)]
    
    return hyperparameter_combinations

def find_best_nn(model, hyperparameters, train_dataset, validation_dataset, test_dataset):
    nn_configs_dict = generate_nn_configs()
    
    train_loader = DataLoader(train_dataset, batch_size = 16, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size = 16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size = 16, shuffle=True)

    for hyperparameters in nn_configs_dict:
        train_start_time = time.time()
        train_rmse_loss, validation_rmse_loss = train_model(model) #add parameters
        train_end_time = time.time()
        model_training_duration = train_end_time - train_start_time

        test_rmse_loss, inference_latency = test_model() #add parameters, does the state dict need adding back in ?

        performance_metrics = {}
        performance_metrics['train_RMSE_loss'] = train_rmse_loss
        performance_metrics['train_R_squared'] = train_r2
        performance_metrics['validation_RMSE_loss'] = validation_rmse_loss
        performance_metrics['validation_R_squared'] = validation_r2 
        performance_metrics['test_RMSE_loss'] = test_rmse_loss
        performance_metrics['test_R_squared'] = test_r2
        performance_metrics['training_duration'] = model_training_duration
        performance_metrics['inference_latency'] = inference_latency

        save_model(model, hyperparameters, performance_metrics)


    #Define a function called find_best_nn which calls this function and then sequentially trains models with each config. 
    # It should save the config used in the hyperparameters.json file for each model trained. Return the model, metrics, and hyperparameters. 
    # Save the best model in a folder

def save_model(model, hyperparameters, metrics):   
    model_folder = datetime.fromtimestamp(datetime.timestamp(datetime.now())).strftime("%d-%m-%Y, %H:%M:%S")
    if not os.path.exists('./models/regression/neural_networks'):
        os.makedirs('./models/regression/neural_networks')

    os.makedirs(f'./models/regression/neural_networks/{model_folder}')
    model_path = f'./models/regression/neural_networks/{model_folder}/model.pt'
    torch.save(model.state_dict, model_path)

    with open(f'./models/regression/neural_networks/{model_folder}/hyperparameters.json', 'w') as f:
        json.dump(hyperparameters, f)
    with open(f'./models/regression/neural_networks/{model_folder}/metrics.json', 'w') as f:
        json.dump(metrics, f)
    with open(f'./models/regression/neural_networks/{model_folder}/model.pt', 'w') as f:
        torch.save(model.state_dict(), f)


if __name__ == "__main__":
    dataset = AirbnbNightlyPriceImageDataset() #Creates an instance of the class
    train_dataset, validation_dataset, test_dataset = split_data(dataset)
    model = NN(get_nn_config())

    find_best_nn(model, generate_nn_configs(), train_dataset, validation_dataset, test_dataset)



    # TODO: check metrics to be tests = RMSE_loss, R2, training duration (can it be done in the train func?), inference latency (in evaluate of train?ß)
    # configure code to run efficiently, should be able to take in several different data types so make it flexible for data entry, can calling all the functions to run the model be made a function such as the evaluate model func?
    



    


   


