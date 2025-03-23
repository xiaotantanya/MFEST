import sys
sys.path.append('./')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import os
from dataset.DMDataset import DMDataset
from util import *
from loss import *
from torch.utils.data import DataLoader

class PPN(nn.Module):
    def __init__(self, input_dim, hidden_dim, stock_num=101, window_size=30, info=''):
        super(PPN, self).__init__()
        self.stock_num = stock_num
        self.window_size = window_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.info = info
        self.linear_proj = nn.Linear(4,4)
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim)
        self.TCCB1 = TCCB(input_c=4, output_c=8, kernel_size=(1,3), stride=1, dilation=(1,2 ** 0), padding=(3-1)*(2 ** 0))
        self.TCCB2 = TCCB(input_c=8, output_c=16, kernel_size=(1,3), stride=1, dilation=(1,2 ** 1), padding=(3-1)*(2 ** 1))
        self.TCCB3 = TCCB(input_c=16, output_c=16, kernel_size=(1,3), stride=1, dilation=(1,2 ** 2), padding=(3-1)*(2 ** 2))
        self.conv4 = nn.Conv2d(16,16, kernel_size=(1,self.window_size), stride=1,bias=False)
        self.bias = torch.nn.Parameter(torch.zeros([1, 1, 1]))
        self.linear_out = nn.Linear(in_features=2 * hidden_dim, out_features=1)

        self.model_name = 'PPN'
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9, weight_decay=5e-8)
        self.return_loss = Return_Loss(commission_ratio=CR)
    
    def forward(self, x):
        # x [B N T P]
        x = rearrange(x, 'B N T P -> B P N T')
        batch_size = x.shape[0]

        x_proj = rearrange(x, 'b f a w -> w b a f')
        x_proj = self.linear_proj(x_proj)
        x_proj = rearrange(x_proj, 'w b a f -> b f a w')

        x_lstm = rearrange(x_proj, 'b f a w -> w b a f')
        # x = rearrange(x, 'b f a w -> b f a w')

        for i in range(x_lstm.shape[2]):
            input_item = x_lstm[:,:,i,:]
            out, (hn, cn) = self.lstm(input_item)
            out = out[-1:, :, :]
            if i==0:
                result = out
            else:
                result = torch.cat((result, out), 0)

        x_lstm = result.view(batch_size, self.stock_num, self.hidden_dim)
        x_lstm = x_lstm.permute(0,2,1).contiguous()
        

        x_tccb = x_proj
        x_tccb = self.TCCB1(x_tccb)
        x_tccb = self.TCCB2(x_tccb)
        x_tccb = self.TCCB3(x_tccb)

        # (128,16,11)
        x_tccb = self.conv4(x_tccb)
        x_tccb = F.relu(x_tccb)
        x_tccb = torch.squeeze(x_tccb, -1)
        
        # (128,16+16+1,11)
        output = torch.cat((x_lstm, x_tccb), 1)
        output = output.permute(0,2,1).contiguous()
        output = self.linear_out(output).squeeze() # [128, 12]
        output = F.softmax(output, dim=-1)
        return output

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


class TCCB(nn.Module):
    def __init__(self, input_c, output_c, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TCCB, self).__init__()
        self.dropout = dropout
        self.input_c = input_c
        self.output_c = output_c
        self.padding = padding
        self.Dconv1 = nn.Conv2d(input_c, output_c, kernel_size, dilation=dilation, stride=stride,bias=False)
        self.Dconv2 = nn.Conv2d(output_c,output_c, kernel_size, dilation=dilation, stride=stride,bias=False)
        self.Cconv = nn.Conv2d(output_c, output_c, (11,1), stride=1, padding=(5,0),bias=False)
        self.Rconv =  nn.Conv2d(input_c, output_c, (1,1), stride=1,bias=False)
    
    def forward(self, x):
        if self.input_c != self.output_c:
            res = self.Rconv(x)
        else:
            res = x
        x=F.pad(x,(self.padding,0,0,0),"constant",value=0)
        x = self.Dconv1(x)
        x = F.dropout(F.relu(x), self.dropout)

        x=F.pad(x,(self.padding,0,0,0),"constant",value=0)
        x = self.Dconv2(x)
        x = F.dropout(F.relu(x), self.dropout)
        
        x = self.Cconv(x)
        x = F.dropout(F.relu(x), self.dropout)
        
        return F.relu(res + x)


if __name__ == '__main__':
    inputs = torch.rand(128,101,30,4).cuda()
    ppn = PPN(input_dim=4,hidden_dim=16, stock_num=101, window_size=30).cuda()
    output = ppn(inputs)
    print(output.shape)
    
