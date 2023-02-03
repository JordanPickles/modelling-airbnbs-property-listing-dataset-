
import pandas as pd
import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader

from tabular_data import load_airbnb


class AirbnbNightlyPriceImageDataset(Dataset):
    def __init__(self):
        super().__init__() 
        self.data = pd.read_csv('./airbnb-property-listings/tabular_data/clean_tabular_data.csv')
        self.features, self.labels = load_airbnb(self.data, 'Price_Night')

    def __getitem__(self, index):
        return torch.tensor(self.features[index]), self.labels[index]

    def __len__(self):
        return len(self.features)


dataset = AirbnbNightlyPriceImageDataset() #Creates an instance of the class
dataset[10]
len(dataset)

# train_loader = DataLoader(dataset, batch_size = 16, shuffle=True)

# def train_loader():
#     for batch in train_loader:
#         print(batch)
#         features, labels = batch
#         print(features.shape)
#         print(labels.shape)



