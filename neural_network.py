
import pandas as pd
import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
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


    return train_dataset, test_dataset, validation_dataset


# Function that trains the model
def train(model, dataloader, nn_config, epochs=10):
    writer = SummaryWriter()
    if nn_config['optimiser'] == 'SGD':
        optimiser = torch.optim.SGD
        optimiser = optimiser(model.parameters(),nn_config['learning_rate'])
    batch_index = 0

    for epoch in range(epochs): # Loops through the dataset a number of times
        for batch in dataloader: # Samples different batches of the data from the data loader
            
            X, y = batch # Sets features and labels from the batch
            X = X.type(torch.float32)
            y = y.type(torch.float32)
                        
            prediction = model(X) 
            loss = F.mse_loss(prediction, y) 
            loss = loss.type(torch.float32)
            loss.backward() # Populates the gradients from the parameters of our model
            print(loss.item())

            optimiser.step() #Optimisation step
            optimiser.zero_grad() # Resets the grad to zero as the grads are no longer useful as they were calculated when the model had different parameters

            writer.add_scalar('loss', loss.item(), batch_index)
            batch_index += 1

    

def evaluate_model(model, train_dataset, test_dataset, validation_dataset, nn_config, epochs=10):
    # Train the model
    dataloader = DataLoader(train_dataset, batch_size = 16, shuffle=True)
    train_start_time = time.time()
    trained_model = train(model, dataloader, nn_config, epochs)
    train_end_time = time.time()
    model_training_duration = train_end_time - train_start_time
    model_datetime = datetime.fromtimestamp(datetime.timestamp(datetime.now())).strftime("%d-%m-%Y, %H:%M:%S")
        
    performance_metrics = {}

    #Train Predictions and Metrics
    X_train = [sample[0] for sample in train_dataset]
    X_train = torch.tensor(X_train, dtype = torch.float32)
    y_train = [sample[1] for sample in train_dataset]
    y_train = torch.tensor(y_train, dtype = torch.float32)
    print(X_train)
    print(y_train)
    prediction_start_time = time.time()
    y_train_pred = model(X_train)
    prediction_end_time = time.time()
    inference_latency = prediction_end_time - prediction_start_time / len(y_train)
    rmse_train = sqrt(mean_squared_error(y_train, y_train_pred))
    r2_train = r2_score(mean_squared_error(y_train, y_train_pred))
    performance_metrics['train_RMSE_loss'] = rmse_train
    performance_metrics['train_R_squared'] = r2_train

    #Validation Predictions and Metrics
    X_validation = [sample[0] for sample in validation_dataset]
    X_validation = torch.tensor(X_validation, dtype = torch.float32)
    y_validation = [sample[1] for sample in validation_dataset]
    y_validation = torch.tensor(y_validation, dtype = torch.float32)
    y_validation_pred = model(X_validation)
    rmse_validation = sqrt(mean_squared_error(y_validation, y_validation_pred))
    r2_validation = r2_score(mean_squared_error(y_validation, y_validation_pred)) 
    performance_metrics['validation_RMSE_loss'] = rmse_validation
    performance_metrics['validation_R_squared'] = r2_validation   


    #Test Predictions and Metrics
    X_test = [sample[0] for sample in test_dataset]
    X_test = torch.tensor(X_test, dtype = torch.float32)
    y_test = [sample[1] for sample in test_dataset]
    y_test = torch.tensor(y_test, dtype = torch.float32)
    y_test_pred = model(X_test)
    rmse_test = sqrt(mean_squared_error(y_test, y_test_pred))
    r2_test = r2_score(mean_squared_error(y_test, y_test_pred))
    performance_metrics['train_RMSE_loss'] = rmse_test
    performance_metrics['train_R_squared'] = r2_test
   

    performance_metrics['training_duration'] = model_training_duration
    performance_metrics['inference_latency'] = inference_latency

    save_model(model, nn_config, performance_metrics, model_datetime)
       

def get_nn_config(config_file = 'nn_config.yaml') -> dict:
    with open(config_file, 'r') as f:
        nn_config = yaml.safe_load(f)
    return nn_config


def save_model(model, hyperparameters, metrics, model_folder):    
    # Ensures directories are created
    if not os.path.exists('./models'):
        os.makedirs('./models')
    if not os.path.exists('./models/regression'):
        os.makedirs('./models/regression')
    if not os.path.exists('./models/regression/neural_networks'):
        os.makedirs('./models/regression/neural_networks')

    with open(f'./models/regression/neural_networks/{model_folder}/hyperparameters.json', 'w') as f:
        json.dump(hyperparameters, f)
    with open(f'./models/regression/neural_networks/{model_folder}/metrics.json', 'w') as f:
        json.dump(metrics, f)
    with open(f'./models/regression/neural_networks/{model_folder}/metrics.pt', 'w') as f:
        torch.save(model.state_dict(), f)


if __name__ == "__main__":

    dataset = AirbnbNightlyPriceImageDataset() #Creates an instance of the class
    train_dataset, test_dataset, validation_dataset = split_data(dataset)

    model = NN(get_nn_config())
    # train(model, DataLoader(train_dataset, batch_size = 16, shuffle=True), get_nn_config())
    # Dataloader for each set


    evaluate_model(model,train_dataset, validation_dataset, test_dataset, get_nn_config())


