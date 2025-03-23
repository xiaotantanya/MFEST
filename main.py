import torch
import gc
import argparse
import pandas as pd
import numpy as np
import time
import json
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from dataset.DataMatrix import DataMatrices
from dataset.DMDataset import DMDataset
from einops import rearrange, repeat
from loss import *
from util import *


def choose_model(model_name, input_dim, info):
    model = None
    if model_name == 'ALSTM':
        from models.ALSTM import ALSTM
        model = ALSTM(d_feat=input_dim, hidden_size=16, info=info)
    elif model_name == 'SFM':
        from models.SFM import SFM
        model = SFM(d_feat=input_dim, hidden_size=16, info=info)
    elif model_name == 'TCN':
        from models.TCN import TCN
        model = TCN(d_feat=input_dim)
    elif model_name == 'DLinear':
        from models.DLinear import DLinear
        model = DLinear(d_feat=input_dim)
    elif model_name == 'GAT':
        from models.GAT import GAT
        model = GAT(d_feat=4, hidden_size=32, num_layers=2, dropout=0.1, info=info)
    elif model_name == 'EIIE':
        from models.EIIE import EIIE
        model = EIIE(in_feature=input_dim, out_feature=1, company_num=101, info=info)
    elif model_name == 'SARL':
        from models.SARL import SARL
        model = SARL(price_dim=input_dim, hidden_dim=16, day=30, layers=1, dropout_rate=0.1, info=info)
    elif model_name == 'AlphaPortfolio':
        from models.AlphaPortfolio import AlphaPortfolio
        model = AlphaPortfolio(info=info)
    elif model_name == 'PPN':
        from models.PPN import PPN
        model = PPN(input_dim=input_dim, hidden_dim=16, stock_num=101, window_size=30, info=info)
    elif model_name == 'RAT':
        from models.RAT import RAT
        model = RAT(in_feature=4, out_feature=1, batch_size=BATCH_SIZE, coin_num=101, info=info)
    elif model_name == 'DeepTrader':
        from models.DeepTrader import DeepTrader
        model = DeepTrader(num_nodes=101, price_in_features=4, hidden_dim=16, window_len=30, dropout=0.1, info=info)
    elif model_name == 'MGCGRU':
        from models.MGCGRU import MGCGRU
        model = MGCGRU(C=4, V=101, T=30, out_feature=1, info=info)
    elif model_name == 'StockMixer':
        from models.StockMixer import StockMixer
        model = StockMixer(101, 30, 4, 32, info='')
    elif model_name == 'MASTER':
        from models.MASTER import MASTER
        model = MASTER(info='')
    elif model_name == 'HSTM':
        from models.HSTM import HSTM
        model = HSTM(input_dim=4, hidden_dim=4, dropout=0.1, local_len=3, stock_num=101, window_size=30, info=info)
    elif model_name == 'HSTGM':
        from models.HSTGM import HSTGM
        model = HSTGM(input_dim=4, hidden_dim=4, dropout=0.1, local_len=3, stock_num=101, window_size=30, info=info)
    elif model_name == 'MFEST_WF':
        from models.MFEST_WF import MFEST_WF
        model = MFEST_WF(input_dim=4, hidden_dim=4, dropout=0.1, local_len=3, stock_num=101, window_size=30, info=info)
    elif model_name == 'MFEST_WIG':
        from models.MFEST_WIG import MFEST_WIG
        model = MFEST_WIG(input_dim=4, hidden_dim=4, dropout=0.1, local_len=3, stock_num=101, window_size=30, info=info)
    elif model_name == 'MFEST_WSG':
        from models.MFEST_WSG import MFEST_WSG
        model = MFEST_WSG(input_dim=4, hidden_dim=4, dropout=0.1, local_len=3, stock_num=101, window_size=30, info=info)
    elif model_name == 'MFEST_WDG':
        from models.MFEST_WDG import MFEST_WDG
        model = MFEST_WDG(input_dim=4, hidden_dim=4, dropout=0.1, local_len=3, stock_num=101, window_size=30, info=info)
    else:
        print("Empty Model")
        exit(1)
    return model.cuda()


def train_common_model(DM, model_name, input_dim=4, stock_num=100, index=0, info=''):
    print(f'{model_name} train {index} th time')
    index = index + 1
    model = choose_model(model_name, input_dim, info)
    model.train_model(DM, index)
    w_list, last_w_list, y_list = model.test()
    w = torch.cat(w_list, dim=0)
    # if w.shape ==:
    w = rearrange(w, '(B N) -> B N', N=stock_num)
    last_w = torch.cat(last_w_list, dim=0)
    # last_w = rearrange(last_w, '(B N) -> B N', N=stock_num)
    y = torch.cat(y_list, dim=0)
    y = rearrange(y, '(B N) -> B N', N=stock_num)

    AR, SR, CR, MDD, r_list = Get_Result(y, w, last_w)

    return_dir = f'./returns/{DM.market}/{model_name}_{info}_{index}.pkl'
    r_list = pd.DataFrame(r_list, columns=['RET']).RET
    r_list.to_pickle(return_dir)

    csv_dir = f'./results/{DM.market}/{model_name}_{info}.csv'
    d = {
        "index": [index],
        "AR": [AR.item()],
        "SR": [SR.item()],
        "CR": [CR.item()],
        "MDD": [MDD.item()],
    }
    new_data_frame = pd.DataFrame(data=d)
    if os.path.isfile(csv_dir):
        dataframe = pd.read_csv(csv_dir)
        dataframe = dataframe.append(new_data_frame)
    else:
        dataframe = new_data_frame
    dataframe.to_csv(csv_dir, index=False)
    DM.reset_PVM()
    model.release_gpu()
    del model
    gc.collect()


if __name__ == '__main__':
    # python main.py --model ALSTM --market ChinaA

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ALSTM')  # Model
    parser.add_argument('--market', type=str, default='ChinaA')  # Dataset
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--device', type=str, default="0")

    info = ''
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    market = args.market  # ChinaA NASDAQ
    model_name = args.model
    index = args.index
    stock_num = 100
    window_size = 30
    input_dim = 4
    train_times = 1
    DM = DataMatrices(market, asset_num=stock_num, window_size=window_size)
    train_common_model(DM, model_name, input_dim, stock_num=stock_num + 1, index=index, info=info)
