
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


    return train_dataset, validation_dataset, test_dataset


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
            
            optimiser.step() #Optimisation step
            optimiser.zero_grad() # Resets the grad to zero as the grads are no longer useful as they were calculated when the model had different parameters

            writer.add_scalar('loss', loss.item(), batch_index)
            batch_index += 1

def model_accuracy(model, dataloader, nn_config, epochs=10):
    for epoch in range(epochs): # Loops through the dataset a number of times
        for batch in dataloader: # Samples different batches of the data from the data loader
            
            X, y = batch # Sets features and labels from the batch
            X = X.type(torch.float32)
            y = y.type(torch.float32)
                        
            prediction = model(X) 
            #add different loss measures per requirement, e.g. rmse, r2 etc
            rmse_loss = F.mse_loss(prediction, y) #TODO work out how to calculate this accurately
            rmse_loss = loss.type(torch.float32) 
            #TODO calculate R2
    return rmse, r2
            

def evaluate_model(model, train_dataset, validation_dataset, test_dataset, nn_config):
    
    train_loader = DataLoader(train_dataset, batch_size = 16, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size = 16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size = 16, shuffle=True)



    # Train the model

    train_start_time = time.time()
    train(model, train_loader, nn_config())
    train_end_time = time.time()
    model_training_duration = train_end_time - train_start_time
    model_datetime = datetime.fromtimestamp(datetime.timestamp(datetime.now())).strftime("%d-%m-%Y, %H:%M:%S")
        
    performance_metrics = {}

    prediction_start_time = time.time()
    prediction_end_time = time.time()
    inference_latency = prediction_end_time - prediction_start_time / len(y_train) # TODO move the prediction times into the train or accuracy function?

    #Evaluating training set  
    train_rmse_losss, train_r2 = model_accuracy(model, train_loader)
    performance_metrics['train_RMSE_loss'] = train_rmse_losss
    performance_metrics['train_R_squared'] = train_r2

    #Evaluating validation set
    validation_rmse_losss, validation_r2 = model_accuracy(model, test_loader)
    performance_metrics['validation_RMSE_loss'] = validation_rmse_losss
    performance_metrics['validation_R_squared'] = validation_r2  


    #Evaluating test set
    test_rmse_losss, test_r2 = model_accuracy(model, validation_loader)
    performance_metrics['test_RMSE_loss'] = test_rmse_losss
    performance_metrics['test_R_squared'] = test_r2
   



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
    train_dataset, validation_dataset, test_dataset = split_data(dataset)

    model = NN(get_nn_config())

    evaluate_model(train_dataset, validation_dataset, test_dataset, get_nn_config())


    # TODO: check metrics to be tests = RMSE_loss, R2, training duration (can it be done in the train func?), inference latency (in evaluate of train?ß)
    # watch last video for saving and loading model (possibly train model, save the state dict and then reload it to test?)
    # should data be batched when evaluating ?
    # configure code to run efficiently, should be able to take in several different data types so make it flexible for data entry, can calling all the functions to run the model be made a function such as the evaluate model func?
    



    


   


