import sys
sys.path.append('./')
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

import torch
import copy
import math
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, LayerNorm

class AlphaPortfolio(nn.Module):
    def __init__(self, info=''):
        super().__init__()
        self.model = ActionNet()
        # for p in self.model.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)

        self.model_name = 'AlphaPortfolio'
        self.info = info
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9, weight_decay=5e-8)
        self.return_loss = Return_Loss(commission_ratio=CR)

    def forward(self, x):
        x = x.unsqueeze(0)
        output = self.model(x)
        return output
    
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
        for i in range(200):
            self.train()
            train_error = self.__train_one_epoch()
            train_loss.append(train_error)
            
            self.eval()
            # valid and early stop
            with torch.no_grad():
                valid_error = self.__valid_one_epoch()
                
            valid_loss.append(valid_error)
            print(f'Epoch {i}: Train Loss: {train_error}, Val loss: {valid_error}')
            if np.isnan(train_error) or np.isnan(valid_error):
                self.load_state_dict(torch.load(f'./saved_models/{self.dataset_name}/{self.model_name}_{self.info}_{self.index}.pt'))
                return train_loss, valid_loss
            if valid_error < min_error:
                print('Save Model!!')
                min_error = valid_error
                no_update_steps = 0
                # save model
                torch.save(self.state_dict(), f'./saved_models/{self.dataset_name}/{self.model_name}_{self.info}_{self.index}.pt')
            else:
                no_update_steps += 1
            
            if no_update_steps > 30: # early stop
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



#attention block
class SelfAttention(nn.Module):
    
    def __init__(self, **kwargs):
        self.drop_rate = 0.1
        super(SelfAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(self.drop_rate)
        
    def forward(self, queries, keys, values): #自注意力
        d = queries.shape[-1] #dk为特征维度，这里为50
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = torch.softmax(scores, dim=2)
        
        return torch.bmm(self.dropout(self.attention_weights), values) #逐元素相乘


#multihead attention block
class MultiHeadAttention(nn.Module):
    
    def __init__(self, config, bias=False):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = config.num_heads
        self.attention = SelfAttention()
        self.W_q = nn.Linear(config.query_dim, config.hidden_size, bias=bias)
        self.W_k = nn.Linear(config.key_dim,config.hidden_size, bias=bias)
        self.W_v = nn.Linear(config.value_dim, config.hidden_size, bias=bias)
        self.W_o = nn.Linear(config.hidden_size, config.hidden_size, bias=bias)

    def forward(self, queries, keys, values):
        
        queries = self.transpose_qkv(self.W_q(queries), self.num_heads)
        keys = self.transpose_qkv(self.W_k(keys), self.num_heads)
        values = self.transpose_qkv(self.W_v(values), self.num_heads)
        output = self.attention(queries, keys, values)
        output_concat = self.transpose_output(output, self.num_heads)
        return self.W_o(output_concat)
    
    def transpose_qkv(self,X, num_heads):
        X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
        X = X.permute(0, 2, 1, 3)
        return X.reshape(-1, X.shape[2], X.shape[3])
    
    def transpose_output(self,X, num_heads):
        X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)


class MLP(nn.Module):
    def __init__(self,key_dim):
        super(MLP, self).__init__()
        self.num_features = key_dim
        self.num_neurons = 32
        self.MLPDropout = 0.1
        self.fc1 = Linear(self.num_features, self.num_neurons) 
        self.fc2 = Linear(self.num_neurons, self.num_features) 
        self.RELU = nn.ReLU()
        self.dropout = Dropout(self.MLPDropout)#dropout rate = 0.1
        self._init_weights() #initializing the weights and bias

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)#全连接层weights初始化，xavier的均匀分布初始化
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)#bias初始化
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.RELU(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TE(nn.Module):
    def __init__(self,config):
        super(TE,self).__init__()
        self.Multihead = MultiHeadAttention(config)
        self.attentionnorm = nn.LayerNorm(config.key_dim, eps=1e-6)
        self.mlp = MLP(config.key_dim)
        self.fcnorm = nn.LayerNorm(config.key_dim, eps=1e-6)
    
    def forward(self,x): #x:2*10*12*50
        a,b,c = x.shape[1],x.shape[2],x.shape[3]
        x = x.reshape(-1,x.shape[2],x.shape[3]) #20*12*50
        h = x
        x = self.Multihead(x,x,x) #20*12*50 
        x = x + h #20*12*50
        x = self.attentionnorm(x)
        
        h = x
        x = self.mlp(x)
        x = x+h 
        # x = self.fcnorm(x) #20*12*50
        x = x.reshape(-1,a,b,c)#2*10*12*50
        x = x.flatten(start_dim = 2) #2*10*600
        
        return x

class CAAN(nn.Module):
    
    def __init__(self,config):
        
        super(CAAN,self).__init__()
        self.W_Q = nn.Linear(config.Query_dim,config.CAAN_dim) #600->50
        self.W_K = nn.Linear(config.Query_dim,config.CAAN_dim) #600->50
        self.W_V = nn.Linear(config.Query_dim,config.CAAN_dim) #600->50
        self.fc1 = nn.Linear(config.CAAN_dim,30) #50->30
        self.fc2 = nn.Linear(30,1) #30->1 #winner score
        self.Tanh = nn.Tanh() #tanh activation function
        self.size = config.long_short_size
        self.fcnorm = nn.LayerNorm(config.num_assets, eps=1e-6)
        
        
    def forward(self,R):
    
        Query_transform = self.W_Q(R) 
        Key_transform = self.W_K(R)
        Value_transform = self.W_V(R)
        dk = Key_transform.shape[-1]
        gamma_matrix = torch.matmul(Query_transform,Key_transform.transpose(1,2)) / math.sqrt(dk) 
        gamma_sum = torch.sum(torch.exp(gamma_matrix),dim = 2).unsqueeze(2)#10*10
        SAAT_matrix = torch.exp(gamma_matrix)/gamma_sum #10*10
        a_matrix = torch.matmul(SAAT_matrix,Value_transform) #10*50
        winner_score = self.fc2(self.fc1(a_matrix)).squeeze(2) #output winner score 2*1000
        # num_assets = winner_score.shape[-1] #每个截面的资产个数
        # long_short_size = self.size #多空资产数量
        # score_rank = torch.sort(winner_score,1,descending = True)[1]+1 #升序排列，取index并+1，最高分索引即为1
        #norm
        # winner_score = self.fcnorm(winner_score) #新发现：加入一个layer norm效果会更好
        # for i in range(winner_score.shape[0]): #T*2*N算法复杂度，T*N
        #     long_sum = 0 
        #     short_sum = 0
        #     for j in range(num_assets):
        #         if 1 <= score_rank[i,j] <= long_short_size: #long portfolio
        #             long_sum += math.exp(winner_score[i,j])
        #         elif num_assets-long_short_size < score_rank[i,j] <= num_assets:
        #             short_sum += math.exp(-1*winner_score[i,j])
        #     for j in range(num_assets):
        #         if 1 <= score_rank[i,j] <= long_short_size: 
        #             winner_score[i,j] = math.exp(winner_score[i,j])/long_sum
        #         elif num_assets-long_short_size < score_rank[i,j] <= num_assets:
        #             winner_score[i,j] = -1*math.exp(-1*winner_score[i,j])/short_sum
        #         else:
        #             winner_score[i,j] = 0     
        return winner_score

import ml_collections
def get_AP_config():
    '''self-defined config here'''
    config = ml_collections.ConfigDict()
    config.key_dim = 4 #个股原始特征的数目
    config.query_dim = 4
    config.value_dim = 4
    config.hidden_size = 4
    config.num_heads = 2
    config.dropout = 0.1 
    config.Query_dim = 120 #展开后的数目 N*F,12*50
    config.CAAN_dim = 50 
    config.num_assets = 101 #各个截面期的资产数目
    config.long_short_size = 10 #多空组合的资产数目
    
    return config

class ActionNet(nn.Module):
    def __init__(self):
        super(ActionNet,self).__init__()
        config = get_AP_config()
        self.Transformer_Encoder = TE(config) 
        self.CAAN_PG = CAAN(config)
        self.fc_out = nn.Linear(32, 1)
        self.softmax = nn.Softmax()
        
    def forward(self,x):
        x = self.Transformer_Encoder(x)
        x = self.CAAN_PG(x).squeeze()
        x = self.softmax(x)
        return x

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    model = ActionNet().cuda()
    x = torch.rand(1, 100,30,4).cuda()
    out = model(x)
    print(out.shape)
