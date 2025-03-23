import os
import sys
sys.path.append('./')
import math
import pandas as pd
import numpy as np
import collections

import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader, TensorDataset
from einops import rearrange, repeat
from dataset.DMDataset import DMDataset
from util import *
from loss import *

class MFEST_WIG(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, dropout=0.5, local_len=3, stock_num=101, window_size=30, info=''):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout 
        self.local_len = local_len
        self.stock_num = stock_num
        self.window_size = window_size
        self.info = info

        self.fstb1= FSTB(input_dim=self.input_dim, hidden_dim=self.hidden_dim, dilation=0, local_len=self.local_len, 
                         dropout=self.dropout, stock_num=self.stock_num, window_size=self.window_size)
        self.fstb2= FSTB(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim, dilation=1, local_len=self.local_len, 
                         dropout=self.dropout, stock_num=self.stock_num, window_size=self.window_size)
        self.fstb3= FSTB(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim, dilation=2, local_len=self.local_len, 
                         dropout=self.dropout, stock_num=self.stock_num, window_size=self.window_size)
        self.fc_in = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc_out = nn.Linear(self.hidden_dim, 1)

        self.model_name = 'MFEST_WIG'
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9, weight_decay=5e-8)
        self.return_loss = Return_Loss(commission_ratio=CR, risk=True)

    def forward(self, x):
        # x[B, N, T, P]
        x = rearrange(x, 'B N T P -> (B N) T P')
        x = self.fstb1(x, self.indus_relation, self.sec_relation)
        x = self.fstb2(x, self.indus_relation, self.sec_relation)
        x = self.fstb3(x, self.indus_relation, self.sec_relation)
        x = x[:,-1,:]
        x = self.fc_out(x)
        x = rearrange(x, '(B N) P -> B N P', N=self.stock_num).squeeze()
        x = F.softmax(x, dim=-1)
        return x


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
        self.indus_relation = torch.Tensor(self.DM.indus_relation).cuda()
        self.sec_relation = torch.Tensor(self.DM.sec_relation).cuda()
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
        # valid_loss = []
        # train_loss = []
        for i in range(MAX_EPOCH):
            self.train()
            train_error = self.__train_one_epoch()
            # train_loss.append(train_error)
            
            self.eval()
            # valid and early stop
            with torch.no_grad():
                valid_error = self.__valid_one_epoch()
                
            # valid_loss.append(valid_error)
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
            torch.cuda.empty_cache()
    
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
    
    def prepare_data(self, DM):
        self.DM = DM
        self.indus_relation = torch.Tensor(self.DM.indus_relation).cuda()
        self.sec_relation = torch.Tensor(self.DM.sec_relation).cuda()
        self.dataset_name = self.DM.market

        train_dataset = DMDataset(DM, DM.train_period, device='cuda', window_size=DM.window_size)
        val_dataset = DMDataset(DM, DM.val_period, device='cuda', window_size=DM.window_size)
        test_dataset = DMDataset(DM, DM.test_period, device='cuda', window_size=DM.window_size)
        self.train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        self.val_dl = DataLoader(val_dataset, batch_size=1, shuffle=False)
        self.test_dl = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    def release_gpu(self):
        if self.train_dl is not None:
            del self.train_dl
        if self.val_dl is not None:
            del self.val_dl
        if self.test_dl is not None:
            del self.test_dl
        torch.cuda.empty_cache()

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):  
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return self.dropout(self.norm(x + sublayer(x)))

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    # input: query, key, value [N T D]
    # output: output_feature [N T D]
    # output: p_attn # [N T T]
    d_k = query.size(-1)  
    scores = torch.matmul(query, key.transpose(-2, -1)) \
        / math.sqrt(d_k) # [N T T]
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)  
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class CTB(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=32, dilation=0, local_len=3, dropout=0.5, stock_num=101, window_size=30):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dilation = dilation
        self.local_len = local_len
        self.dropout = dropout
        self.stock_num = stock_num
        self.window_size = window_size
        self.q_linear = nn.Linear(input_dim, hidden_dim)
        self.k_linear = nn.Linear(input_dim, hidden_dim)
        self.v_linear = nn.Linear(input_dim, hidden_dim)

        self.padding = (local_len-1)*(2 ** dilation)
        self.k_cov = nn.Conv2d(stock_num, stock_num, (local_len, 1), dilation=(2 ** dilation, 1), stride=1, bias=False)
        self.v_cov = nn.Conv2d(stock_num, stock_num, (local_len, 1), dilation=(2 ** dilation, 1), stride=1, bias=False)
    
    def forward(self, input):
        # [(B N) T D]
        bn =input.shape[0]
        q = self.q_linear(input)
        k = self.k_linear(input)
        v = self.v_linear(input)

        k = F.pad(k,(0,0,self.padding,0),"constant",value=0)
        k = rearrange(k, '(B N) T D -> B N T D', N=self.stock_num)
        k = self.k_cov(k)
        k = rearrange(k, 'B N T D -> (B N) T D')

        v = F.pad(v,(0,0,self.padding,0),"constant",value=0)
        v = rearrange(v, '(B N) T D -> B N T D', N=self.stock_num)
        v = self.v_cov(v)
        v = rearrange(v, 'B N T D -> (B N) T D')
        out, _ = attention(q, k, v)
        return out


class FSTB(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=32, dilation=0, local_len=3, dropout=0.5, stock_num=101, window_size=30):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dilation = dilation
        self.local_len = local_len
        self.dropout = dropout
        self.stock_num = stock_num
        self.window_size = window_size
        self.dominance_freq = 12

        self.input_fc = nn.Linear(self.input_dim, self.hidden_dim)

        # self.freq_upsampler = nn.ModuleList()
        # for i in range(self.input_dim):
        #     self.freq_upsampler.append(nn.Linear(self.dominance_freq, self.dominance_freq).to(torch.cfloat))
        self.freq_upsampler = nn.Linear(self.dominance_freq, self.dominance_freq).to(torch.cfloat)

        self.ctb = CTB(input_dim=hidden_dim, hidden_dim=hidden_dim, dilation=dilation
                       , local_len=local_len, dropout=dropout, stock_num=stock_num, window_size=window_size)
        self.sub_layer1 = SublayerConnection(self.hidden_dim, self.dropout)

        self.gcn = GraphConvolution(self.hidden_dim,self.hidden_dim)

        self.output_fc = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.sub_layer2 = SublayerConnection(self.hidden_dim, self.dropout)
        
    
    def forward(self, x, ind_adj, sec_adj):
        # x[B, N, T, P]
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var=torch.var(x, dim=1, keepdim=True)+ 1e-5
        x = x / torch.sqrt(x_var)
        
        low_specx = torch.fft.rfft(x, dim=1)
        low_specx[:,self.dominance_freq:]=0 # LPF
        low_specx = low_specx[:,0:self.dominance_freq,:] # LPF
        
        # for i in range(self.input_dim):
        #     low_specx[:,:,i]=self.freq_upsampler[i](low_specx[:,:,i].permute(0,1)).permute(0,1)
        low_specxy_ = self.freq_upsampler(low_specx.permute(0,2,1)).permute(0,2,1)
        
        low_specxy = torch.zeros([low_specx.size(0),int((self.window_size)/2+1),low_specx.size(2)],dtype=low_specx.dtype).to(low_specx.device)
        low_specxy[:,0:low_specxy_.size(1),:]=low_specxy_
        low_xy=torch.fft.irfft(low_specxy, dim=1)
        x_low=(low_xy) * torch.sqrt(x_var) + x_mean

        x_f = self.input_fc(x_low)
        x_f = self.sub_layer1(x_f, self.ctb)
        x_f = self.gcn(x_f, ind_adj, sec_adj)
        x_f = self.sub_layer2(x_f, self.output_fc)
        return x_f


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, stock_num=101):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.stock_num = stock_num
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.dy_adj = Parameter(torch.FloatTensor(stock_num, stock_num))
        self.reset_parameters_kaiming()

    def reset_parameters_kaiming(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')
        nn.init.kaiming_normal_(self.dy_adj.data, a=0, mode='fan_in')

    def forward(self, x, ind_adj, sec_adj):
        # x [B N T P]
        # adj [N N]
        x = rearrange(x, '(B N) T P -> B N T P', N=self.stock_num)
        # dgr = torch.diag(ind_adj.sum(1)**-0.5)
        # y = torch.matmul(dgr, ind_adj)
        # ind_adj = torch.matmul(y, dgr)

        # dgr = torch.diag(sec_adj.sum(1)**-0.5)
        # y = torch.matmul(dgr, sec_adj)
        # ind_adj = torch.matmul(y, dgr)
        
        y = torch.matmul(x, self.weight)
        y = rearrange(y, 'B N T P -> B P T N')
        # y = torch.matmul(y, ind_adj)
        y = torch.matmul(y, sec_adj)
        y = torch.matmul(y, self.dy_adj)
        y = rearrange(y, 'B P T N -> (B N) T P')
        return y


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    model = HSTGM(input_dim=4, hidden_dim=32, dropout=0.1, local_len=3, stock_num=101, window_size=30).cuda()
    # model = FSTB(input_dim=4, hidden_dim=32, dropout=0.5, stock_num=101).cuda()
    # model = CTB(input_dim=4, hidden_dim=32, dilation=2, local_len=3, dropout=0.5, stock_num=101, window_size=30).cuda()
    x = torch.rand([16, 101, 30, 4]).cuda()
    print(model(x).shape)

