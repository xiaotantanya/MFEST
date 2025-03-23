import os
import math
import pandas as pd
import numpy as np
import collections

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from einops import rearrange, repeat
from dataset.DMDataset import DMDataset
from util import *
from loss import *


class ALSTM(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, rnn_type="GRU", info=''):
        super().__init__()
        
        self.hid_size = hidden_size
        self.input_size = d_feat
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.rnn_layer = num_layers
        self._build_model()
        
        self.model_name = 'ALSTM'
        self.info = info
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9, weight_decay=5e-8)
        self.return_loss = Return_Loss(commission_ratio=CR)

    def _build_model(self):
        try:
            klass = getattr(nn, self.rnn_type.upper())
        except Exception as e:
            raise ValueError("unknown rnn_type `%s`" % self.rnn_type) from e
        self.net = nn.Sequential()
        self.net.add_module("fc_in", nn.Linear(in_features=self.input_size, out_features=self.hid_size))
        self.net.add_module("act", nn.Tanh())
        self.rnn = klass(
            input_size=self.hid_size,
            hidden_size=self.hid_size,
            num_layers=self.rnn_layer,
            batch_first=True,
            dropout=self.dropout,
        )
        # self.fc_transfor = nn.Linear(in_features=self.hid_size * 2, out_features=self.hid_size)
        self.fc_out = nn.Linear(in_features=self.hid_size * 2, out_features=1)
        self.att_net = nn.Sequential()
        self.att_net.add_module(
            "att_fc_in",
            nn.Linear(in_features=self.hid_size, out_features=int(self.hid_size / 2)),
        )
        self.att_net.add_module("att_dropout", torch.nn.Dropout(self.dropout))
        self.att_net.add_module("att_act", nn.Tanh())
        self.att_net.add_module(
            "att_fc_out",
            nn.Linear(in_features=int(self.hid_size / 2), out_features=1, bias=False),
        )
        self.att_net.add_module("att_softmax", nn.Softmax(dim=1))
        self.softmax = nn.Softmax()

        self.BN = nn.BatchNorm1d(30)


    def forward(self, inputs):
        # inputs = self.BN(inputs)
        rnn_out, _ = self.rnn(self.net(inputs))  # [batch, seq_len, num_directions * hidden_size]
        attention_score = self.att_net(rnn_out)  # [batch, seq_len, 1]
        out_att = torch.mul(rnn_out, attention_score)
        out_att = torch.sum(out_att, dim=1)
        out = self.fc_out(
            torch.cat((rnn_out[:, -1, :], out_att), dim=1)
        )  # [batch, seq_len, num_directions * hidden_size] -> [batch, 1]

        out = self.softmax(out.squeeze())
        return out

    def __train_one_epoch(self):
        epoch_loss = 0.0
        for i, (x, y, idx) in enumerate(self.train_dl):
            self.optimizer.zero_grad()

            w = self.forward(x.squeeze(0))
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
            w = self.forward(x.squeeze(0))
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
        self.train_dl = DataLoader(train_dataset, batch_size=1, shuffle=True)
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
            w = self.forward(x.squeeze(0))
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