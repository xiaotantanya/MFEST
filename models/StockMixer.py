import sys
sys.path.append('./')
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import math
import pandas as pd
import numpy as np
import collections

import torch
from torch import nn
from einops import rearrange, repeat
from torch.nn.utils import weight_norm
from torch.utils.data import Dataset, DataLoader, TensorDataset
from einops import rearrange, repeat
from dataset.DMDataset import DMDataset
from util import *
from loss import *

acv = nn.GELU()

def get_loss(prediction, ground_truth, base_price, mask, batch_size, alpha):
    device = prediction.device
    all_one = torch.ones(batch_size, 1, dtype=torch.float32).to(device)
    return_ratio = torch.div(torch.sub(prediction, base_price), base_price)
    reg_loss = F.mse_loss(return_ratio * mask, ground_truth * mask)
    pre_pw_dif = torch.sub(
        return_ratio @ all_one.t(),
        all_one @ return_ratio.t()
    )
    gt_pw_dif = torch.sub(
        all_one @ ground_truth.t(),
        ground_truth @ all_one.t()
    )
    mask_pw = mask @ mask.t()
    rank_loss = torch.mean(
        F.relu(pre_pw_dif * gt_pw_dif * mask_pw)
    )
    loss = reg_loss + alpha * rank_loss
    return loss, reg_loss, rank_loss, return_ratio


class MixerBlock(nn.Module):
    def __init__(self, mlp_dim, hidden_dim, dropout=0.0):
        super(MixerBlock, self).__init__()
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        self.dense_1 = nn.Linear(mlp_dim, hidden_dim)
        self.LN = acv
        self.dense_2 = nn.Linear(hidden_dim, mlp_dim)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.LN(x)
        if self.dropout != 0.0:
            x = F.dropout(x, p=self.dropout)
        x = self.dense_2(x)
        if self.dropout != 0.0:
            x = F.dropout(x, p=self.dropout)
        return x


class Mixer2d(nn.Module):
    def __init__(self, time_steps, channels):
        super(Mixer2d, self).__init__()
        self.LN_1 = nn.LayerNorm([time_steps, channels])
        self.LN_2 = nn.LayerNorm([time_steps, channels])
        self.timeMixer = MixerBlock(time_steps, time_steps)
        self.channelMixer = MixerBlock(channels, channels)

    def forward(self, inputs):
        x = self.LN_1(inputs)
        x = x.permute(0, 2, 1)
        x = self.timeMixer(x)
        x = x.permute(0, 2, 1)

        x = self.LN_2(x + inputs)
        y = self.channelMixer(x)
        return x + y


class TriU(nn.Module):
    def __init__(self, time_step):
        super(TriU, self).__init__()
        self.time_step = time_step
        self.triU = nn.ParameterList(
            [
                nn.Linear(i + 1, 1)
                for i in range(time_step)
            ]
        )

    def forward(self, inputs):
        x = self.triU[0](inputs[:, :, 0].unsqueeze(-1))
        for i in range(1, self.time_step):
            x = torch.cat([x, self.triU[i](inputs[:, :, 0:i + 1])], dim=-1)
        return x


class TimeMixerBlock(nn.Module):
    def __init__(self, time_step):
        super(TimeMixerBlock, self).__init__()
        self.time_step = time_step
        self.dense_1 = TriU(time_step)
        self.LN = acv
        self.dense_2 = TriU(time_step)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.LN(x)
        x = self.dense_2(x)
        return x


class MultiScaleTimeMixer(nn.Module):
    def __init__(self, time_step, channel, scale_count=1):
        super(MultiScaleTimeMixer, self).__init__()
        self.time_step = time_step
        self.scale_count = scale_count
        self.mix_layer = nn.ParameterList([nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=2 ** i, stride=2 ** i),
            TriU(int(time_step / 2 ** i)),
            nn.Hardswish(),
            TriU(int(time_step / 2 ** i))
        ) for i in range(scale_count)])
        self.mix_layer[0] = nn.Sequential(
            nn.LayerNorm([time_step, channel]),
            TriU(int(time_step)),
            nn.Hardswish(),
            TriU(int(time_step))
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        y = self.mix_layer[0](x)
        for i in range(1, self.scale_count):
            y = torch.cat((y, self.mix_layer[i](x)), dim=-1)
        return y


class Mixer2dTriU(nn.Module):
    def __init__(self, time_steps, channels):
        super(Mixer2dTriU, self).__init__()
        self.LN_1 = nn.LayerNorm([time_steps, channels])
        self.LN_2 = nn.LayerNorm([time_steps, channels])
        self.timeMixer = TriU(time_steps)
        self.channelMixer = MixerBlock(channels, channels)

    def forward(self, inputs):
        x = self.LN_1(inputs)
        x = x.permute(0, 2, 1)
        x = self.timeMixer(x)
        x = x.permute(0, 2, 1)

        x = self.LN_2(x + inputs)
        y = self.channelMixer(x)
        return x + y


class MultTime2dMixer(nn.Module):
    def __init__(self, time_step, channel, scale_dim=8):
        super(MultTime2dMixer, self).__init__()
        self.mix_layer = Mixer2dTriU(time_step, channel)
        self.scale_mix_layer = Mixer2dTriU(scale_dim, channel)

    def forward(self, inputs, y):
        y = self.scale_mix_layer(y)
        x = self.mix_layer(inputs)
        return torch.cat([inputs, x, y], dim=1)


class NoGraphMixer(nn.Module):
    def __init__(self, stocks, hidden_dim=20):
        super(NoGraphMixer, self).__init__()
        self.dense1 = nn.Linear(stocks, hidden_dim)
        self.activation = nn.Hardswish()
        self.dense2 = nn.Linear(hidden_dim, stocks)
        self.layer_norm_stock = nn.LayerNorm(stocks)

    def forward(self, inputs):
        x = inputs
        x = x.permute(1, 0)
        x = self.layer_norm_stock(x)
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)
        x = x.permute(1, 0)
        return x


class StockMixer(nn.Module):
    def __init__(self, stocks, time_steps, channels, market, info=''):
        super(StockMixer, self).__init__()
        scale_dim = 15
        self.mixer = MultTime2dMixer(time_steps, channels, scale_dim=scale_dim)
        self.channel_fc = nn.Linear(channels, 1)
        self.time_fc = nn.Linear(time_steps * 2 + scale_dim, 1)
        self.conv = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=2, stride=2)
        self.stock_mixer = NoGraphMixer(stocks, market)
        self.time_fc_ = nn.Linear(time_steps * 2 + scale_dim, 1)
        self.softmax = nn.Softmax()

        self.model_name = 'StockMixer'
        self.info = info
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9, weight_decay=5e-8)
        self.return_loss = Return_Loss(commission_ratio=CR)

    def forward(self, inputs):
        # inputs N T D
        x = inputs.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        y = self.mixer(inputs, x)
        y = self.channel_fc(y).squeeze(-1)

        z = self.stock_mixer(y)
        y = self.time_fc(y)
        z = self.time_fc_(z)
        output = y + z
        output = self.softmax(output.squeeze())
        return output
    
    def __train_one_epoch(self):
        epoch_loss = 0.0
        for i, (x, y, idx) in enumerate(self.train_dl):
            self.optimizer.zero_grad()

            w = self.forward(x.squeeze(0))
            last_w = self.DM.get_w(idx)
            self.DM.set_w(w, idx)
            loss = self.return_loss(y, w.unsqueeze(0), last_w)
            if torch.isnan(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.parameters(), 0.3)
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(self.train_dl)

    def __valid_one_epoch(self):
        val_loss = 0.0
        for i, (x, y, idx) in enumerate(self.val_dl):
            w = self.forward(x.squeeze(0))
            last_w = self.DM.get_w(idx)
            self.DM.set_w(w, idx)
            loss = self.return_loss(y, w.unsqueeze(0), last_w)
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
            self.load_state_dict(torch.load(f'./saved_models/{self.dataset_name}/{self.model_name}_{self.info}_{self.index}.pt'))
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

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    model = StockMixer(101, 30, 4, 32, info='').cuda()
    inputs = torch.rand([101,30,4]).cuda()
    output = model(inputs)
    print(output.shape)
