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


class MGCGRU(nn.Module):
    def __init__(self, C=4, V=101, T = 30, out_feature=1, info=''):
        super(MGCGRU, self).__init__()
        self.dyn = nn.Parameter(torch.FloatTensor(V, V), requires_grad=True) # 空间维度上一个动态的图卷积
        self.dim = 64
        self.GCNs = nn.ModuleList([
            nn.Linear(C, 64),
            # nn.Linear(32, 64),
            nn.Linear(64, self.dim),
        ])

        self.GRUs = nn.ModuleList([
            nn.ModuleDict({
                "GRU_z": nn.Linear(C + self.dim + self.dim, self.dim),
                "GRU_r": nn.Linear(C + self.dim + self.dim, self.dim),
                "GRU_w": nn.Linear(C + self.dim + self.dim, self.dim),
            }),
            # nn.ModuleDict({
            #     "GRU_z": nn.Linear(self.dim + self.dim + self.dim, self.dim),
            #     "GRU_r": nn.Linear(self.dim + self.dim + self.dim, self.dim),
            #     "GRU_w": nn.Linear(self.dim + self.dim + self.dim, self.dim),
            # }),
            # nn.ModuleDict({
            #     "GRU_z": nn.Linear(self.dim + self.dim + self.dim, self.dim),
            #     "GRU_r": nn.Linear(self.dim + self.dim + self.dim, self.dim),
            #     "GRU_w": nn.Linear(self.dim + self.dim + self.dim, self.dim),
            # }),
        ])

        self.fc = self.out = nn.Sequential(
            nn.Linear(self.dim * V, 128),
            nn.ReLU(),
            nn.Linear(128, out_feature * V),
        )
        self.softmax = nn.Softmax()

        self.info = info
        self.model_name = 'MGCGRU'
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9, weight_decay=5e-8)
        self.return_loss = Return_Loss(commission_ratio=CR)

        # todo: 发现不加这个会出现 Nan
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.dyn.shape[1])
        self.dyn.data.uniform_(-stdv, stdv)


    def forward(self, inputs):
        '''

        :param inputs: b, c, v, t
        :return: b, 1/2, v, 1
        '''
        inputs = rearrange(inputs, 'B N T P -> B P N T')
        b, c, v, t = inputs.shape
        inputs = inputs.permute(0, 3, 2, 1).contiguous() # [b, t, v, c]

        h_gru = [] # [b, v, 128]
        for gru_idx in range(len(self.GRUs)):
            h = torch.zeros(b, v, self.dim, dtype=inputs.dtype, device=inputs.device)
            h_gru.append(h)

        for t in range(t):
            # GCN
            feature_curr = inputs[:, t, :, :] # [b, v, c]
            for gcn_idx in range(len(self.GCNs)):
                feature_curr = torch.matmul(self.dyn, feature_curr) # [b, v, c]
                feature_curr = torch.tanh(self.GCNs[gcn_idx](feature_curr))# [b, v, d1] # [b, 40, 128]

            xt = inputs[:, t, :, :]
            for gru_idx in range(len(h_gru)):
                # GRU
                z = torch.sigmoid(self.GRUs[gru_idx]["GRU_z"](torch.cat([xt, feature_curr, h_gru[gru_idx]], dim=-1)))  # b, v, 128
                r = torch.sigmoid(self.GRUs[gru_idx]["GRU_r"](torch.cat([xt, feature_curr, h_gru[gru_idx]], dim=-1)))  # b, v, 128
                c = torch.tanh(self.GRUs[gru_idx]["GRU_w"](torch.cat([xt, feature_curr, torch.multiply(r, h_gru[gru_idx])], dim=-1))) # [b, v, 128]
                ht = torch.multiply((1 - z), h_gru[gru_idx]) + torch.multiply(z, c)  # b, 15, 16
                xt = ht
                h_gru[gru_idx] = ht

        out = self.fc(ht.view(b, -1))
        out = self.softmax(out)
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
            w_list.append(w.squeeze(0))
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
    model = MGCGRU(C=4, V=101, T = 30, out_feature=1).cuda()
    price = torch.randn((16, 101, 30, 4)).cuda()
    out = model(price)
    print(out.shape)
