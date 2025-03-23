import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import time

import time
from datetime import datetime

# class Return_Loss(nn.Module):
#     def __init__(self, commission_ratio=0.003, risk=False):
#         super(Return_Loss, self).__init__()
#         self.commission_ratio = commission_ratio
#         self.risk = risk
#         # self.max_weight = 0.25
#         # self.m = 0.001
#         # self.relu = nn.ReLU()

#     def forward(self, y, w, last_w):   
#         # y[batch, N]  w[batch, N]      
#         element_reward = w * y  # [batch, asset + 1]
#         r_list = torch.sum(element_reward, dim=1)  # [batch]
#         # pure_r = 1-torch.sum(torch.abs(w-last_w), 1) * self.commission_ratio # [batch]
#         # pure_r = pure_r.cuda() 
#         # r_list = r_list * pure_r

#         loss = r_list.sum()

#         if self.risk:
#             loss_risk = (r_list * r_list / 2).sum()
#             loss = loss - 0.1 * loss_risk
#         return -loss


class Return_Loss(nn.Module):
    def __init__(self, commission_ratio=0.003, risk=False, risk_weight=0.6):
        super(Return_Loss, self).__init__()
        self.commission_ratio = commission_ratio
        self.risk = risk
        self.risk_weight = risk_weight
        # self.max_weight = 0.25
        # self.m = 0.001
        # self.relu = nn.ReLU()

    def forward(self, y, w, last_w):   
        # y[batch, N]  w[batch, N]      
        element_reward = w * y  # [batch, asset + 1]
        r_list = torch.sum(element_reward, dim=1)  # [batch]
        # pure_r = 1-torch.sum(torch.abs(w-last_w), 1) * self.commission_ratio # [batch]
        # pure_r = pure_r.cuda() 
        # r_list = r_list * pure_r

        loss = r_list.sum()

        if self.risk:
            loss_risk = (r_list * r_list / 2).sum()
            loss = loss - self.risk_weight * loss_risk
        return -loss


def Get_Result(y, w, last_w, commission_ratio=0.003):
    element_reward = w * y  # [batch, asset + 1]
    r_list = torch.sum(element_reward, dim=1)  # [batch]
    # pure_r = 1-torch.sum(torch.abs(w-last_w), 1) * self.commission_ratio # [batch]
    # pure_r = pure_r.cuda() 
    # r_list = r_list * pure_r
    # r_list = r_list.cpu().numpy()

    AR = r_list.mean() * 256
    AV = r_list.std() * np.sqrt(256)
    SR = AR / AV
    MDD = max_drawdown(r_list)
    CR = AR / MDD
    r_list = r_list.detach().cpu().numpy().tolist()
    return AR, SR, CR, MDD, r_list

def max_drawdown(r_list):
    portfolio_values = []
    drawdown_list = []
    max_benefit = 0
    for i in range(r_list.shape[0]):
        if i > 0:
            portfolio_values.append(portfolio_values[i - 1] + r_list[i])
        else:
            portfolio_values.append(r_list[i])
        if portfolio_values[i] > max_benefit:
            max_benefit = portfolio_values[i]
            drawdown_list.append(0.0)
        else:
            drawdown_list.append(max_benefit - portfolio_values[i])
    return max(drawdown_list) 
