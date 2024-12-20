import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

class StockDataset(Dataset):
    def __init__(self, data, sequence_length=50):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.sequence_length]
        y = self.data[idx+self.sequence_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def load_and_preprocess_data(file_path, test_size=0.2, val_size=0.1, sequence_length=50):
    df = pd.read_csv(file_path)
    prices = df['CloseUSD'].values.reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    prices_scaled = scaler.fit_transform(prices).flatten()

    train_data, test_data = train_test_split(prices_scaled, test_size=test_size, shuffle=False)
    train_data, val_data = train_test_split(train_data, test_size=val_size / (1 - test_size), shuffle=False)

    train_dataset = StockDataset(train_data, sequence_length)
    val_dataset = StockDataset(val_data, sequence_length)
    test_dataset = StockDataset(test_data, sequence_length)

    return train_dataset, val_dataset, test_dataset, scaler
