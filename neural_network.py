
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
import shutil
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

        self.y = self.y.view(-1,1)
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


def train_model(train_loader, validation_loader, nn_config, epochs=10):
    model = NN(nn_config)
    writer = SummaryWriter()
    if nn_config['optimiser'] == 'SGD':
        optimiser = torch.optim.SGD
        optimiser = optimiser(model.parameters(), nn_config['learning_rate'])
    if nn_config['optimiser'] == 'Adam':
        optimiser = torch.optim.Adam
        optimiser = optimiser(model.parameters(), nn_config['learning_rate'] )
    batch_index = 0
    train_rmse_loss = 0.0
    validation_rmse_loss = 0.0
    train_r2 = 0.0
    validation_r2 = 0.0

    train_start_time = time.time()
    for epoch in range(epochs): # Loops through the dataset a number of times

        for batch in train_loader: # Samples different batches of the data from the data loader
            
            X_train, y_train = batch # Sets features and labels from the batch
            X_train = X_train.type(torch.float32)
            y_train = y_train.type(torch.float32)
            y_train = y_train.view(-1, 1)
                        
            train_prediction = model(X_train) 
            mse_loss = F.mse_loss(train_prediction, y_train) 
            mse_loss = mse_loss.type(torch.float32)
            mse_loss.backward() # Populates the gradients from the parameters of our model
            
            optimiser.step() #Optimisation step
            optimiser.zero_grad() # Resets the grad to zero as the grads are no longer useful as they were calculated when the model had different parameters

            writer.add_scalar('loss', mse_loss.item(), batch_index)
            batch_index += 1
            rmse_loss = torch.sqrt(mse_loss)
            train_rmse_loss += rmse_loss.item()            

            train_prediction_detached = train_prediction.detach().numpy()
            y_train_detached = y_train.detach().numpy()
            train_r2 += r2_score(y_train_detached, train_prediction_detached)



        for batch in validation_loader: # Samples different batches of the data from the data loader
            
            X_validation, y_validation = batch # Sets features and labels from the batch
            X_validation = X_validation.type(torch.float32)
            y_validation = y_validation.type(torch.float32)
            y_validation = y_validation.view(-1, 1)
                        
            validation_prediction = model(X_validation) 
            mse_loss = F.mse_loss(validation_prediction, y_validation) 
            mse_loss = mse_loss.type(torch.float32) 
            rmse_loss = torch.sqrt(mse_loss)
            validation_rmse_loss += rmse_loss.item()

            #Calculate r2
            validation_prediction_detached = validation_prediction.detach().numpy()
            y_validation_detached = y_validation.detach().numpy()
            validation_r2 += r2_score(y_validation_detached, validation_prediction_detached)


    #Normalises performance metrics to the number of samples passed through the model
    train_rmse_loss = train_rmse_loss/(epochs*len(X_train))
    validation_rmse_loss = validation_rmse_loss/(epochs*len(X_train))
    train_r2 = train_r2 / (epochs*len(X_train))
    validation_r2 = validation_r2 / (epochs*len(X_train))

    model_name = datetime.fromtimestamp(datetime.timestamp(datetime.now())).strftime("%d-%m-%Y, %H:%M:%S")

    if not os.path.exists('./models/regression/neural_networks/trained_models'):
        os.makedirs('./models/regression/neural_networks/trained_models')
    model_path = f'./models/regression/neural_networks/trained_models/{model_name}'
    torch.save(model.state_dict(), model_path)

    return train_rmse_loss, validation_rmse_loss, train_r2, validation_r2, model_name


def test_model(nn_config, state_dict_path, dataloader):
    model = NN(nn_config)
    test_rmse_loss = 0.0
    test_r2 = 0.0
    
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)

    for batch in dataloader: # Samples different batches of the data from the data loader
        
        X_test, y_test = batch # Sets features and labels from the batch
        X_test = X_test.type(torch.float32)
        y_test = y_test.type(torch.float32)
        y_test = y_test.view(-1, 1)
       

        prediction_start_time = time.time()
        test_prediction = model(X_test) 
        inference_latency = (time.time() - prediction_start_time) / len(test_prediction)    
     
        #Calculate MSE Loss
        mse_loss = F.mse_loss(test_prediction, y_test)
        mse_loss = mse_loss.type(torch.float32) 

        #Calculate RMSE loss
        rmse_loss = torch.sqrt(mse_loss)
        test_rmse_loss += rmse_loss.item()

        #Calculate r2
        prediction_detached = test_prediction.detach().numpy()
        y_test_detached = y_test.detach().numpy()
        test_r2 += r2_score(y_test_detached, prediction_detached)
    
    test_rmse_loss = test_rmse_loss/(len(X_test))
    test_r2 = test_r2 / len(X_test)

    return test_rmse_loss, inference_latency, test_r2
           

def get_nn_config(config_file = 'nn_config.yaml') -> dict:
    with open(config_file, 'r') as f:
        nn_config = yaml.safe_load(f)
    return nn_config

def generate_nn_configs():
    hyperparameters = {
        'optimiser': ['SGD', 'Adam'],
        'learning_rate': [0.0001, 0.001, 0.01],
        'hidden_layer_width': [12],
        'model_depth': [3, 4] 
    }
    keys = hyperparameters.keys()
    values = hyperparameters.values()
    hyperparameter_combinations = [dict(zip(keys, combination)) for combination in product(*values)]
    
    return hyperparameter_combinations

def find_best_nn(hyperparameters, train_dataset, validation_dataset, test_dataset):
    nn_configs_dict = generate_nn_configs()
    
    train_loader = DataLoader(train_dataset, batch_size = 16, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size = 16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size = 16, shuffle=True)

    best_rmse_loss = np.inf
    best_model_name = None
    best_model_path = None
    best_hyperparameters = None
    performance_metrics = {}


    for nn_config in nn_configs_dict:
        
        train_start_time = time.time()
        train_rmse_loss, validation_rmse_loss, train_r2, validation_r2, model_name = train_model(train_loader, validation_loader, nn_config) 
        model_training_duration = time.time() - train_start_time
      
        # Logic applied to decide if model trained is better than the previous best model, if so the best parameters are updated
        if validation_rmse_loss < best_rmse_loss:
            best_rmse_loss = validation_rmse_loss
            best_model_name = model_name
            best_model_path = f'./models/regression/neural_networks/trained_models/{best_model_name}'
            best_hyperparameters = nn_config
            best_model_metrics = performance_metrics
            test_rmse_loss, inference_latency, test_r2 = test_model(best_hyperparameters, best_model_path, test_loader) 
            #Adds perfromance metrics to dict
            performance_metrics['train_RMSE_loss'] = train_rmse_loss
            performance_metrics['train_R_squared'] = train_r2
            performance_metrics['validation_RMSE_loss'] = validation_rmse_loss
            performance_metrics['validation_R_squared'] = validation_r2 
            performance_metrics['test_RMSE_loss'] = test_rmse_loss
            performance_metrics['test_R_squared'] = test_r2
            performance_metrics['training_duration'] = model_training_duration
            performance_metrics['inference_latency'] = inference_latency
    
            
    return best_model_name, best_model_path, best_hyperparameters, best_model_metrics


def save_model(model_name, model_path, hyperparameters, metrics):

    # Creates a folder for the best model trained and then moves model state dict into a folder specifically for that model   
    new_model_folder = os.makedirs(f'./models/regression/neural_networks/{model_name}')
    new_model_path = f'./models/regression/neural_networks/{model_name}/model.pt'
    shutil.move(model_path, new_model_path)


    #Saves hyperparameters and model performance metrics
    with open(f'./models/regression/neural_networks/{model_name}/hyperparameters.json', 'w') as f:
        json.dump(hyperparameters, f)
    with open(f'./models/regression/neural_networks/{model_name}/metrics.json', 'w') as f:
        json.dump(metrics, f)



if __name__ == "__main__":
    dataset = AirbnbNightlyPriceImageDataset() #Creates an instance of the class
    train_dataset, validation_dataset, test_dataset = split_data(dataset)
    best_model_name, best_model_path, best_hyperparameters, performance_metrics  = find_best_nn(generate_nn_configs(),train_dataset, validation_dataset, test_dataset)
    save_model(best_model_name, best_model_path, best_hyperparameters, performance_metrics)


    #TODO: scale the data in split data?
    



    


   


