import sys
import os
sys.path.append('./')
from einops import rearrange
from dataset.DMDataset import DMDataset
from util import *
from loss import *
from torch.utils.data import DataLoader
import math
import torch
import torch.nn as nn


class DeepTrader(nn.Module):
    def __init__(self, num_nodes=101, price_in_features=4, hidden_dim=32, window_len=30,
                 dropout=0.1, info=''):
        super().__init__()

        self.asu = AssetScoreUnit(num_nodes=num_nodes, in_features=price_in_features, hidden_dim=hidden_dim, window_len=window_len,
                 dropout=dropout, kernel_size=2, layers=5, supports=None,
                 spatial_bool=True, addaptiveadj=True, aptinit=None)

        self.msu = MarketScoreUnit(in_features=hidden_dim, window_len=window_len, hidden_dim=hidden_dim)

        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_bn = nn.BatchNorm1d(hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax()

        self.info = info
        self.model_name = 'DeepTrader'
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9, weight_decay=5e-8)
        self.return_loss = Return_Loss(commission_ratio=CR)



    def forward(self, price):
        '''
        # 用 news 当做市场消息
        Args:
            price:
            news: 

        Returns:
            B N
        '''

        price_feature = self.asu(price) # [8, 40, 128]


        volatility_predicted = self.out_linear(price_feature)
        volatility_predicted = torch.tanh(self.out_bn(volatility_predicted.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous())
        volatility_predicted = self.out(volatility_predicted)

        volatility_predicted = volatility_predicted.squeeze()
        volatility_predicted = self.softmax(volatility_predicted)
        return volatility_predicted

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


class AssetScoreUnit(nn.Module):
    def __init__(self, num_nodes=30, in_features=6, hidden_dim=128, window_len=13,
                 dropout=0.3, kernel_size=2, layers=4, supports=None,
                 spatial_bool=True, addaptiveadj=True, aptinit=None):
        super(AssetScoreUnit, self).__init__()
        self.hidden_dim = hidden_dim

        self.sagcn = SAGCN(num_nodes, in_features, hidden_dim, window_len, dropout, kernel_size, layers,
                           supports, spatial_bool, addaptiveadj, aptinit)
        self.bn1 = nn.BatchNorm1d(num_features=num_nodes)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)

        # self.in1 = nn.InstanceNorm1d(num_features=num_nodes)
        # self.lstm = nn.LSTM(input_size=in_features, hidden_size=hidden_dim, )

    def forward(self, inputs):
        """
        inputs: [batch, num_stock, window_len, num_features]
        outputs: [batch, scores]
        """

        x = self.bn1(self.sagcn(inputs))
        out = self.linear1(x) # .squeeze(-1) # [b, 30]
        # out = 1 / ((-out).exp() + 1)
        return out


class MarketScoreUnit(nn.Module):
    def __init__(self, in_features=4, window_len=13, hidden_dim=128):
        super(MarketScoreUnit, self).__init__()
        self.in_features = in_features
        self.window_len = window_len
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_size=in_features, hidden_size=hidden_dim)
        self.attn1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.attn2 = nn.Linear(hidden_dim, 1)

        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        # self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, X):
        """
        :X: [batch_size(B), window_len(L), in_features(I)]
        :return: Parameters: [batch, 2]
        """
        X = X.permute(1, 0, 2) # [30, b, 4]

        outputs, (h_n, c_n) = self.lstm(X)  # [13, b, 128], [1, b, 128], [1, b, 128]
        H_n = h_n.repeat((self.window_len, 1, 1)) # [13, b, 128]
        scores = self.attn2(torch.tanh(self.attn1(torch.cat([outputs, H_n], dim=2))))  # [13, B, 1]
        scores = scores.squeeze(2).transpose(1, 0)  # [B, 13]
        attn_weights = torch.softmax(scores, dim=1)
        outputs = outputs.permute(1, 0, 2)  # [B, 13, 128]
        attn_embed = torch.bmm(attn_weights.unsqueeze(1), outputs).squeeze(1) # [b, 1, 13] * [b, 13, 128] -> [b, 128]
        out = torch.tanh(self.bn1(self.linear1(attn_embed))) # [b, 128] # todo: 把 relu 改成了 tanh
        # out = self.linear2(out) # [b, 2]  # mu, sigma
        return out



class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class GraphConvNet(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=2, order=2):
        super(GraphConvNet, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = nn.functional.dropout(h, self.dropout, training=self.training)
        return h

class SpatialAttentionLayer(nn.Module):
    def __init__(self, num_nodes, in_features, in_len):
        super(SpatialAttentionLayer, self).__init__()
        self.W1 = nn.Linear(in_len, 1, bias=False)
        self.W2 = nn.Linear(in_features, in_len, bias=False)
        self.W3 = nn.Linear(in_features, 1, bias=False)
        self.V = nn.Linear(num_nodes, num_nodes)

        self.bn_w1 = nn.BatchNorm1d(num_features=num_nodes)
        self.bn_w3 = nn.BatchNorm1d(num_features=num_nodes)
        self.bn_w2 = nn.BatchNorm1d(num_features=num_nodes)

    def forward(self, inputs):
        # inputs: (batch, num_features, num_nodes, window_len)
        part1 = inputs.permute(0, 2, 1, 3)
        part2 = inputs.permute(0, 2, 3, 1)
        part1 = self.bn_w1(self.W1(part1).squeeze(-1))
        part1 = self.bn_w2(self.W2(part1))
        part2 = self.bn_w3(self.W3(part2).squeeze(-1)).permute(0, 2, 1)  #
        S = torch.softmax(self.V(torch.relu(torch.bmm(part1, part2))), dim=-1)
        return S



class SAGCN(nn.Module):
    def __init__(self, num_nodes=30, in_features=6, hidden_dim=128, window_len=13,
                 dropout=0.3, kernel_size=2, layers=4, supports=None,
                 spatial_bool=True, addaptiveadj=True, aptinit=None):

        super(SAGCN, self).__init__()
        self.dropout = dropout
        self.layers = layers
        if spatial_bool:
            self.gcn_bool = True
            self.spatialattn_bool = True
        else:
            self.gcn_bool = False
            self.spatialattn_bool = False

        self.addaptiveadj = addaptiveadj

        self.tcns = nn.ModuleList()
        self.gcns = nn.ModuleList()
        self.sans = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.supports = supports

        self.start_conv = nn.Conv2d(in_features, hidden_dim, kernel_size=(1, 1))

        self.bn_start = nn.BatchNorm2d(hidden_dim)

        receptive_field = 1
        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if self.gcn_bool and addaptiveadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec = nn.Parameter(torch.randn(num_nodes, 1), requires_grad=True)
                self.supports_len += 1

            else:
                raise NotImplementedError

        additional_scope = kernel_size - 1
        a_s_records = []
        dilation = 1
        for l in range(layers):
            tcn_sequence = nn.Sequential(nn.Conv2d(in_channels=hidden_dim,
                                                   out_channels=hidden_dim,
                                                   kernel_size=(1, kernel_size),
                                                   dilation=dilation),
                                         nn.ReLU(),
                                         nn.Dropout(dropout),
                                         nn.BatchNorm2d(hidden_dim))

            self.tcns.append(tcn_sequence)

            self.residual_convs.append(nn.Conv2d(in_channels=hidden_dim,
                                                 out_channels=hidden_dim,
                                                 kernel_size=(1, 1)))

            self.bns.append(nn.BatchNorm2d(hidden_dim))

            if self.gcn_bool:
                self.gcns.append(GraphConvNet(hidden_dim, hidden_dim, dropout, support_len=self.supports_len))

            dilation *= 2
            a_s_records.append(additional_scope)
            receptive_field += additional_scope
            additional_scope *= 2

        self.receptive_field = receptive_field
        if self.spatialattn_bool:
            for i in range(layers):
                self.sans.append(SpatialAttentionLayer(num_nodes, hidden_dim, receptive_field - a_s_records[i]))
                receptive_field -= a_s_records[i]

    def forward(self, X):
        '''

        Args:
            X: [batch, num_stock, window_len, num_features]

        Returns:

        '''
        batch = X.shape[0]
        X = X.permute(0, 3, 1, 2)  # [b, 6, 30, 13] [batch, feature, stocks, length]
        in_len = X.shape[3]
        if in_len < self.receptive_field:
            x = nn.functional.pad(X, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = X
        assert not torch.isnan(x).any()
        # x = rearrange(X, 'B P N T -> (B P) N T')
        x = self.bn_start(self.start_conv(x)) # [b, 128, 30, 16]
        # x = rearrange(X, '(B P) N T -> B P N T', B=batch)
        new_supports = None
        if self.gcn_bool and self.addaptiveadj and self.supports is not None:
            adp_matrix = torch.softmax(torch.relu(torch.mm(self.nodevec, self.nodevec.t())), dim=0)
            new_supports = self.supports + [adp_matrix]

        for i in range(self.layers):
            residual = self.residual_convs[i](x)
            x = self.tcns[i](x) # [b, 128, 40, 16] -> [15] -> [13] -> [9] -> [1]
            if self.gcn_bool and self.supports is not None:
                if self.addaptiveadj:
                    x = self.gcns[i](x, new_supports)
                else:
                    x = self.gcns[i](x, self.supports)

            if self.spatialattn_bool:
                attn_weights = self.sans[i](x)
                x = torch.einsum('bnm, bfml->bfnl', (attn_weights, x))

            x = x + residual[:, :, :, -x.shape[3]:]

            x = self.bns[i](x)

        # (batch, num_nodes, hidden_dim)
        return x.squeeze(-1).permute(0, 2, 1)

if __name__ == "__main__":
    model = DeepTrader(num_nodes=101, price_in_features=4, hidden_dim=32, window_len=30,
                 dropout=0.1).cuda()
    price = torch.randn((16, 101, 30, 4)).cuda()
    out = model(price)
    print(out.shape)