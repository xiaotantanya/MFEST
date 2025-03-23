import sys
sys.path.append('./')
import pandas as pd
import numpy as np
import json

def get_industry_relation(selected_tickers, industry_ticker_file):
    ticker_index = {}
    for index, ticker in enumerate(selected_tickers):
        ticker_index[ticker] = index
    with open(industry_ticker_file, 'r') as fin:
        industry_tickers = json.load(fin)
    ticker_relation_embedding = np.zeros([len(selected_tickers), len(selected_tickers)], dtype=int)
    for industry in industry_tickers.keys():
        cur_ind_tickers = industry_tickers[industry]
        for i in range(len(cur_ind_tickers)):
            if cur_ind_tickers[i] not in selected_tickers:
                continue
            target_tickers_ind = ticker_index[cur_ind_tickers[i]]
            for j in range(len(cur_ind_tickers)):
                if cur_ind_tickers[j] not in selected_tickers:
                    continue
                item_tickers_ind = ticker_index[cur_ind_tickers[j]]
                ticker_relation_embedding[target_tickers_ind][item_tickers_ind] = 1
    return ticker_relation_embedding


def get_code2id_dic(path):
    df = pd.read_csv(path, encoding='gbk')
    choose_df = df[df['Id']!='unknown']
    code = list(choose_df['Code'])
    id = list(choose_df['Id'])
    code2id ={}
    for i in range(len(code)):
        code2id[code[i]] = id[i]
    return code2id

def get_wiki_relation(selected_tickers, connection_file, code2id_file):
    code2id = get_code2id_dic(code2id_file)

    ticker_index = {}
    for index, ticker in enumerate(selected_tickers):
        ticker_index[ticker] = index

    with open(connection_file, 'r') as fin:
        connections = json.load(fin)
    wiki_relation_dic = {}
    for i in range(len(selected_tickers)):
        if selected_tickers[i] in code2id.keys():
            target_id = code2id[selected_tickers[i]]
        else:
            continue
        if target_id not in connections.keys():
            continue
        target_relation_dic = connections[target_id]
        target_tickers_ind = ticker_index[selected_tickers[i]]
        for j in range(len(selected_tickers)):
            if selected_tickers[j] in code2id.keys():
                item_id = code2id[selected_tickers[j]]
            else:
                continue
            if item_id not in target_relation_dic.keys():
                continue
            item_relation_list = target_relation_dic[item_id]
            item_tickers_ind = ticker_index[selected_tickers[j]]
            for k in range(len(item_relation_list)):
                if len(item_relation_list[k]) == 1:
                    relation_name = item_relation_list[k][0]
                else:
                    relation_name = item_relation_list[k][0] + '_' + item_relation_list[k][1]
                if relation_name not in wiki_relation_dic.keys():
                    wiki_relation_dic[relation_name] = np.eye(N=len(selected_tickers), M=len(selected_tickers), dtype=int)
                wiki_relation_dic[relation_name][target_tickers_ind][item_tickers_ind] = 1
    wiki_relation_embedding = []
    for key in wiki_relation_dic.keys():
        item_relation = wiki_relation_dic[key]
        wiki_relation_embedding.append(item_relation)
    wiki_relation_embedding = np.array(wiki_relation_embedding)
    return wiki_relation_embedding


if __name__ == '__main__':
    df = pd.read_csv('./database/relation_data/ChinaA/ChinaA_wiki_id.csv',encoding='gbk')
    choose_df = df[df['Id']!='unknown']
    code = list(choose_df['Code'])

    # industry_ticker_file = './database/relation_data/ChinaA/ChinaA_industry_ticker.json'
    industry_ticker_file = './database/relation_data/ChinaA/ChinaA_industry.json'
    ticker_relation_embedding = get_industry_relation(code, industry_ticker_file)
    print(ticker_relation_embedding.shape)

    sector_ticker_file = './database/relation_data/ChinaA/ChinaA_sector.json'
    sector_relation_embedding = get_industry_relation(code, sector_ticker_file)
    print(sector_relation_embedding.shape)

    connection_file = './database/relation_data/ChinaA/ChinaA_connections.json'
    code2id_file = './database/relation_data/ChinaA/ChinaA_wiki_id.csv'
    wiki_relation_embedding = get_wiki_relation(code, connection_file, code2id_file)
    print(wiki_relation_embedding.shape)