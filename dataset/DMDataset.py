import sys
sys.path.append('./')
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
import numpy as np
from einops import rearrange, repeat
import pandas as pd
from dataset.DataMatrix import DataMatrices

class DMDataset(Dataset):
    def __init__(self, DM, period, device='cuda', window_size=30):
        super(DMDataset, self).__init__()
        self.DM = DM 
        self.global_data = DM.global_data # [N T P]
        self.window_size = window_size
        self.period = period # [start, end]
        self.device = device
        self.__prepare_data()
        
    
    def __prepare_data(self):
        self.panel = self.global_data[:, self.period[0]:self.period[1]+2,:]
        self.date_len = self.panel.shape[1] - self.window_size
        
    def __getitem__(self, idx):
        x = self.panel[:,idx:idx+self.window_size,:] / self.panel[:,idx+self.window_size-1:idx+self.window_size,-1:]# [N T P]
        y = self.panel[:, idx + self.window_size, -1] / self.panel[:, idx + self.window_size - 1, -1] - 1
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)
        idx = torch.tensor(idx, dtype=torch.int).to(self.device)
        return x, y, idx
    
    def __len__(self):
        return self.date_len
    

if __name__ == '__main__':
    market = 'NASDAQ'
    DM = DataMatrices(market, asset_num=100)
    dataset = DMDataset(DM, DM.test_period, device='cuda', window_size=30)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

    for (x, y, idx) in dataloader:
        print(idx)