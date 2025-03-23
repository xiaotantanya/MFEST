from datetime import datetime
import pandas as pd
import numpy as np
import math
import time
import os
import torch
import sys
from dataset.Relation import get_industry_relation, get_wiki_relation
import warnings
warnings.filterwarnings("ignore")


class DataMatrices:
    def __init__(self, market, asset_num=100, window_size=30):
        self.asset_num = asset_num
        self.window_size = window_size
        self.market = market
        self.asset_list = self.select_asset() 
    
        self.global_data = self.get_global_panel() # [P, N, T]
        self.global_data = self.global_data.transpose((1,2,0)) # [N, T, P]
        self.date_number = self.global_data.shape[1]
        print(f'{market} Dataset length: {self.date_number}')

        # [date, asset + 1]
        self.__PVM = np.ones((self.global_data.shape[1], self.global_data.shape[0]))/(self.global_data.shape[0])
        self.divide_data()
        self.industry_relation()
        self.sector_relation()
    
    def reset_PVM(self):
        self.__PVM = np.ones((self.global_data.shape[1], self.global_data.shape[0]))/(self.global_data.shape[0])
    
    def set_w(self, w, idx):
        w = w.detach().cpu().numpy()
        idx = idx.cpu().numpy() + 1
        idx = idx.tolist()
        self.__PVM[idx, :] = w
    
    def get_w(self, idx):
        idx = idx.cpu().numpy().tolist()
        return torch.tensor(self.__PVM[idx, :], dtype=torch.float32).cuda()
    
    def get_global_panel(self):
        root_path = '/home/ml_group/liuhaoxian/Financial/theses_code/MFEST/database/' + self.market
        file_list = [f for f in os.listdir(root_path) if os.path.isfile(os.path.join(root_path, f))]
        global_data = []
        for i in range(len(self.asset_list)):
            item_path = root_path + '/' + self.asset_list[i] + '.csv'
            item_data = pd.read_csv(item_path)[['OPEN','HIGH','LOW','CLOSE']].values
            global_data.append(item_data)
        global_data = np.array(global_data)
        global_data = global_data.transpose((2,0,1))
        global_data = np.pad(global_data,((0,0), (1,0),(0,0)),'constant', constant_values=(1,1)) # 加现金
        return global_data
    
    def select_asset(self):
        root_path = '/home/ml_group/liuhaoxian/Financial/theses_code/MFEST/database/' + self.market
        file_list = [f for f in os.listdir(root_path) if os.path.isfile(os.path.join(root_path, f))]
        
        # 选择时间完整的股票,并计算累积交易量
        acc_volumn_dic = {}
        for i in range(len(file_list)):
            item_path = root_path + '/' + file_list[i]
            item_asset = file_list[i][:-4]
            item_df = pd.read_csv(item_path, error_bad_lines=False)
            # item_df = pd.read_csv(item_path, error_bad_lines=False)
            date_number = item_df.shape[0]
            item_df.drop_duplicates(subset=['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME'], keep='first', inplace=True)
            item_volumn = item_df['VOLUME'].values.sum()
            if item_df.shape[0] == date_number:
                acc_volumn_dic[item_asset] = item_volumn
        
        # 根据累计交易量筛选股票
        final_asset_list = []
        acc_volumn_order = sorted(acc_volumn_dic.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
        # print(len(acc_volumn_order))
        assert self.asset_num <= len(acc_volumn_order), 'asset_num is too large'
        for i in range(self.asset_num):
            final_asset_list.append(acc_volumn_order[i][0])
        return final_asset_list

    def divide_data(self):
        date_number = self.date_number
        self.train_period = [0, int(date_number*0.6)]
        self.val_period = [int(date_number*0.6)+1, int(date_number*0.8)]
        self.test_period = [int(date_number*0.8)+1, self.date_number-1]
        print("Train Period:", self.train_period)
        print("Val Period:", self.val_period)
        print("Test Period:", self.test_period)
    
    def industry_relation(self):
        industry_ticker_file = '/home/ml_group/liuhaoxian/Financial/theses_code/MFEST/database/relation_data/' + self.market +'/' + self.market + '_industry.json'
        asset_list = self.asset_list[:]
        if self.market == 'NASDAQ':
            for i in range(len(asset_list)):
                asset_list[i] = asset_list[i]
        ticker_relation_embedding = get_industry_relation(asset_list, industry_ticker_file) # [asset_num, asset_num]
        ticker_relation_embedding = np.pad(ticker_relation_embedding,((1,0),(1,0)),'constant', constant_values=(0,0))
        ticker_relation_embedding[0][0] = 1

        used_list = [0] * ticker_relation_embedding.shape[0]
        class_list = []
        for i in range(ticker_relation_embedding.shape[0]):
          if used_list[i] == 1:
            continue
          item_class = [i]
          for j in range(i+1, ticker_relation_embedding.shape[1]):
            if ticker_relation_embedding[i][j] != 0:
              used_list[j] = 1
              item_class.append(j)
          class_list.append(item_class)
        self.indus_relation = ticker_relation_embedding
    
    def sector_relation(self):
        sector_file = '/home/ml_group/liuhaoxian/Financial/theses_code/MFEST/database/relation_data/' + self.market +'/' + self.market + '_sector.json'
        asset_list = self.asset_list[:]
        if self.market == 'NASDAQ':
            for i in range(len(asset_list)):
                asset_list[i] = asset_list[i]
        ticker_relation_embedding = get_industry_relation(asset_list, sector_file) # [asset_num, asset_num]
        
        ticker_relation_embedding = np.pad(ticker_relation_embedding,((1,0),(1,0)),'constant', constant_values=(0,0))
        ticker_relation_embedding[0][0] = 1

        used_list = [0] * ticker_relation_embedding.shape[0]
        class_list = []
        for i in range(ticker_relation_embedding.shape[0]):
          if used_list[i] == 1:
            continue
          item_class = [i]
          for j in range(i+1, ticker_relation_embedding.shape[1]):
            if ticker_relation_embedding[i][j] != 0:
              used_list[j] = 1
              item_class.append(j)
          class_list.append(item_class)
        self.sec_relation = ticker_relation_embedding


if __name__ == '__main__':
    market = 'NASDAQ'
    batch_size = 1
    window_size = 30
    asset_num = 100
    DM = DataMatrices(market, asset_num, window_size)
    print(DM.indus_relation)
    print(DM.sec_relation)
    


        
