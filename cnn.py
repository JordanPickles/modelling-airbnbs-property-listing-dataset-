
import pandas as pd
import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from tabular_data import load_airbnb


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


class LinearRegression(torch.nn.Module):
    def __init__(self):
        super().__init__() # initalises the parent class
        self.linear_layer = torch.nn.Linear(12,1)

    def forward(self, features):
        return self.linear_layer(features) # Makes prediction

class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super().__init__() # initalises the parent class
        self.linear_layer = torch.nn.Linear(12,1)

    def forward(self, features):
        return F.sigmoid(self.linear_layer(features)) # Makes prediction as a probability between 0 and 1

class NN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #define layers
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(12, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16,1)
        )

    def forward(self, X):
        return self.layers(X)

def split_data(dataset):
    # Splits data into 70% training and 30% test
    train_dataset, test_dataset = random_split(dataset, [int(len(dataset) * 0.7), len(dataset)-int(len(dataset)*0.7)])

    # Splits test data in half, percentage of total dataset is 15% test and 15% validation
    validation_dataset, test_dataset = random_split(test_dataset, [int(len(test_dataset) * 0.5), len(test_dataset)-int(len(test_dataset)*0.5)])

    print(f"    Training: {len(train_dataset)}")
    print(f"    Validation: {len(validation_dataset)}")
    print(f"    Testing: {len(test_dataset)}")

    return train_dataset, test_dataset, validation_dataset


# Function that trains the model
def train(model, dataloader, epochs=10):
    writer = SummaryWriter()
    optimiser = torch.optim.SGD(model.parameters(),lr = 0.001)
    batch_index = 0

    for epoch in range(epochs): # Loops through the dataset a number of times
        for batch in train_loader: # Samples different batches of the data from the data loader
            features, labels = batch # Sets features and labels from the batch
            features = features.type(torch.float32)
            labels = labels.type(torch.float32)
            labels = labels.unsqueeze(1) #Adds new dimension to tensor to ensure the labels and predictions are the same shape

            
            prediction = model(features) # Provides prediction through the linear regression model
            loss = F.mse_loss(prediction, labels) #For linear Regression use F.binary_cross_entropy() instead
            loss = loss.type(torch.float32) # Error for the linear regression
            loss.backward() # Populates the gradients from the parameters of our model
            print(loss.item())

            optimiser.step() #Optimisation step
            optimiser.zero_grad() # Resets the grad to zero as the grads are no longer useful as they were calculated when the model had different parameters

            writer.add_scalar('loss', loss.item(), batch_index)
            batch_index += 1



if __name__ == "__main__":

    dataset = AirbnbNightlyPriceImageDataset() #Creates an instance of the class
    train_dataset, test_dataset, validation_dataset = split_data(dataset)

    model = NN()
    # Dataloaders for each set
    train_loader = DataLoader(train_dataset, batch_size = 16, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size = 16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size = 16, shuffle=True)
    train(model, train_loader)
    train(model, validation_loader)