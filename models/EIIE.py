#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
sys.path.append('./')
from einops import rearrange
from dataset.DMDataset import DMDataset
from util import *
from loss import *
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

class EIIE(nn.Module):
    def __init__(self, in_feature=4, out_feature=1, company_num=101, info=''):
        super(EIIE, self).__init__()

        self.company_num = company_num
        self.day_num = 30
        self.out_feature = out_feature
        self.info = info

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_feature, out_channels=32, kernel_size=5, padding=2), # [b, 32, 40, 30]
            nn.Tanh(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2), # [b, 64, 40, 30]
            nn.Tanh(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, padding=2), # [b, 128, 40, 30]
            nn.Tanh(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, padding=2), # [b, 64, 40, 30]
            nn.Tanh(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, padding=2),  # [b, 64, 40, 30]
            nn.Tanh(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, padding=2),  # [b, 64, 40, 30]
            nn.Tanh(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, padding=2),  # [b, 64, 40, 30]
            nn.Tanh(),
        )
        # # self.out = nn.Conv2d(8, out_feature, kernel_size=(1, 1))
        self.out = nn.Sequential(
            nn.Linear(16 * self.company_num * self.day_num, 128),
            nn.ReLU(),
            # nn.Linear(256, 128),
            # nn.Tanh(),
            nn.Linear(128, out_feature * self.company_num),
        )

        self.model_name = 'EIIE'
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9, weight_decay=5e-8)
        self.return_loss = Return_Loss(commission_ratio=CR)


    def forward(self,x):
        '''

        :param x: b, c, v, t: [b, 17, 40, 30]
        :return:
        '''
        x = rearrange(x, 'B N T P -> B P N T')
        b, c, v, t = x.shape
        x = self.block(x) # [b, c, 40, 1]
        # x = self.out(x)
        x = self.out(x.reshape(b, -1)).squeeze()
        out = F.softmax(x, dim=-1)
        return out
    
    def __train_one_epoch(self):
        epoch_loss = 0.0
        for i, (x, y, idx) in enumerate(self.train_dl):
            self.optimizer.zero_grad()

            w = self.forward(x)
            last_w = self.DM.get_w(idx)
            self.DM.set_w(w, idx)
            loss = self.return_loss(y, w, last_w)
            
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(self.train_dl)

    def __valid_one_epoch(self):
        val_loss = 0.0
        for i, (x, y, idx) in enumerate(self.val_dl):
            w = self.forward(x)
            last_w = self.DM.get_w(idx)
            self.DM.set_w(w, idx)
            loss = self.return_loss(y, w, last_w)
            val_loss += loss.item()
        return val_loss / len(self.val_dl)
    
    def train_model(self, DM, index):
        if 'saved_models' not in os.listdir('./'):
            os.mkdir('saved_models')

        self.DM = DM
        self.dataset_name = self.DM.market
        self.index = index

        train_dataset = DMDataset(DM, DM.train_period, device='cuda', window_size=DM.window_size)
        val_dataset = DMDataset(DM, DM.val_period, device='cuda', window_size=DM.window_size)
        test_dataset = DMDataset(DM, DM.test_period, device='cuda', window_size=DM.window_size)
        self.train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        self.val_dl = DataLoader(val_dataset, batch_size=1, shuffle=False)
        self.test_dl = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        min_error = np.Inf
        no_update_steps = 0
        valid_loss = []
        train_loss = []
        for i in range(MAX_EPOCH):
            self.train()
            train_error = self.__train_one_epoch()
            train_loss.append(train_error)
            
            self.eval()
            # valid and early stop
            with torch.no_grad():
                valid_error = self.__valid_one_epoch()
                
            valid_loss.append(valid_error)
            print(f'Epoch {i}: Train Loss: {train_error}, Val loss: {valid_error}')
            if valid_error < min_error:
                print('Save Model!!')
                min_error = valid_error
                no_update_steps = 0
                # save model
                torch.save(self.state_dict(), f'./saved_models/{self.dataset_name}/{self.model_name}_{self.info}_{self.index}.pt')
            else:
                no_update_steps += 1
            
            if no_update_steps > UPDATE_STEP: # early stop
                print(f'Early stop at epoch {i}')
                break
            # self.load_state_dict(torch.load(f'./saved_models/{self.dataset_name}/{self.model_name}_{self.info}_{self.index}.pt'))
        return train_loss, valid_loss
    
    def test(self):
        self.load_state_dict(torch.load(f'./saved_models/{self.dataset_name}/{self.model_name}_{self.info}_{self.index}.pt'))
        self.eval()
        w_list = []
        last_w_list = []
        y_list = []
        for i, (x, y, idx) in enumerate(self.test_dl):
            w = self.forward(x)
            last_w = self.DM.get_w(idx)
            self.DM.set_w(w, idx)
            w_list.append(w)
            last_w_list.append(last_w)
            y_list.append(y.squeeze(0))
        return w_list, last_w_list, y_list
    
    def release_gpu(self):
        if self.train_dl is not None:
            del self.train_dl
        if self.val_dl is not None:
            del self.val_dl
        if self.test_dl is not None:
            del self.test_dl
        torch.cuda.empty_cache()

if __name__ == '__main__':
    m = EIIE().cuda()
    price_x = torch.randn((4, 101, 30, 4)).cuda()
    out = m(price_x)  # [8, 1, 40, 1]
    print(out.shape)