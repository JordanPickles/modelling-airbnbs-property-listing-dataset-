
import pandas as pd
import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader, random_split

from tabular_data import load_airbnb


class AirbnbNightlyPriceImageDataset(Dataset):
    def __init__(self):
        super().__init__() 
        self.data = pd.read_csv('./airbnb-property-listings/tabular_data/clean_tabular_data.csv')
        self.features, self.labels = load_airbnb(self.data, 'Price_Night')
        assert len(self.features) == len(self.labels) # Data and labels have to be of equal length

    def __getitem__(self, index):
        return torch.tensor(self.features[index]), self.labels[index]

    def __len__(self):
        return len(self.features)


dataset = AirbnbNightlyPriceImageDataset() #Creates an instance of the class
print(dataset[10])
print(len(dataset))

# Splits data into 70% training and 30% test
train_dataset, test_dataset = random_split(dataset, [int(len(dataset) * 0.7), len(dataset)-int(len(dataset)*0.7)])

# Splits test data in half, percentage of total dataset is 15% test and 15% validation
validation_dataset, test_dataset = random_split(test_dataset, [int(len(test_dataset) * 0.5), len(test_dataset)-int(len(test_dataset)*0.5)])

print(f"    Training: {len(train_dataset)}")
print(f"    Validation: {len(validation_dataset)}")
print(f"    Testing: {len(test_dataset)}")

train_loader = DataLoader(dataset, batch_size = 64, shuffle=True)





