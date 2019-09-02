import argparse
import hashlib
from hashlib import *
from imblearn.over_sampling import SMOTE
import numpy as np
import os
import pandas as pd
import pickle
from sklearn import metrics
import nsml
from nsml import DATASET_PATH

pd.options.mode.chained_assignment = None


def preprocess(*, item=[], label=[], mode='train'):
    """
    前処理されたデータを返す

    input: データのファイル名
           label_file (option): 指定したときは、train で呼び出されたとみなす。
    output: pd.dataframe

    やっていること ...
        アンダーサンプリング
        ↓
        各種特徴量抽出・生成
        ↓
        オーバーサンプリング (SMOTE)
    """

    print('### start preprocess ###')

    print('### mode: {} ###\n'.format(mode))

    # 画像の特徴量の取得(train と test でファイルが異なる)
    if mode == 'train' or mode == 'validation':
        feature_file = os.path.join(DATASET_PATH, 'train', 'train_data', 'train_image_features.pkl')
    if mode == 'test':
        feature_file = os.path.join(DATASET_PATH, 'test', 'test_data', 'test_image_features.pkl')
    f = open(feature_file, 'rb')
    dict_feature = pickle.load(f)

    # article データの取得
    if mode == 'train' or 'validation':
        article_file = os.path.join(DATASET_PATH, 'train', 'train_data', 'train_data_article.tsv')
    if mode == 'test':
        article_file = os.path.join(DATASET_PATH, 'test', 'test_data', 'test_data_article.tsv')
    article = pd.read_csv(article_file, sep='\t')

    if mode == 'train':
        print('データの個数: \n', label['label'].value_counts())

    if mode == 'train':
        under = 2
        print('\n### under sampling ( 0の個数が、1の個数の {} 倍 ) ###\n'.format(under))
        item_0 = item[label['label'] == 0]
        item_1 = item[label['label'] == 1]
        item_0 = item_0.sample(n=int(under * len(item_1)))

        ### for debug ###
        # item_0 = item_0.sample(n=10000)
        # item_1 = item_0.sample(n=10000)
        #################

        item = pd.concat([item_0, item_1])
        label_list = [0 for i in range(len(item_0))] + [1 for i in range(len(item_1))]
        label = pd.DataFrame({'label': label_list})
        print('データの個数: \n', label['label'].value_counts())

    item_old = item[['hh', 'article_id', 'read_article_ids', 'age_range']]
    item = item[['hh', 'age_range']]
    article = article.set_index('article_id')
    dict_category = article['category_id'].to_dict()

    print('category : article_id のカテゴリ(整数値)')

    def rep_category_int(t):
        if t in dict_category:
            return dict_category[t]
        else:
            return 13
    item['category'] = item_old['article_id'].map(rep_category_int)

    print('hh : 時間帯')

    print('age_range : 年齢層')
    dict_age_range = {'unknown': 0, '-14': 1, '15-19': 2, '20-24': 3, '25-29': 4, '30-34': 5, '35-39': 6, '40-44': 7, '45-49': 8, '50-': 9}
    item['age_range'] = item['age_range'].replace(dict_age_range)
    item['age_range'] = item['age_range'].astype(int)

    print('read_num : read_article_ids の個数')

    def read_num(t):
        if type(t) == str:
            return len(t.split(','))
        else:
            return 0
    item['read_num'] = item_old['read_article_ids'].map(read_num)
    item['read_num'] = item['read_num'].astype(int)

    ##### read_num のカテゴリー化 #####
    def rep_category_int(t):
        if t in dict_category:
            return dict_category[t]
        else:
            return 0

    def create_read_num_cat(t, i):
        if type(t) != str:
            return 0
        else:
            t_split = t.split(',')
            t_split_cat = list(map(rep_category_int, t_split))
            return t_split_cat.count(i)
    for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 17, 18, 20, 22, 23, 24, 26, 27]:
        column_new = 'read_num_cat{}'.format(i)
        print('{} : read_article_ids の個数 (カテゴリ {})'.format(column_new, i))
        item[column_new] = item_old['read_article_ids'].map(lambda t: create_read_num_cat(t, i))

    for i in range(12):
        # article_id の i 文字目を追加
        column_new = 'id_{}'.format(i + 1)
        print('{} : article_id の {} 文字目'.format(column_new, i + 1))
        item[column_new] = item_old['article_id'].map(lambda t: int(t[i], 16))
        item[column_new] = item[column_new].astype(int)

    def create_history_category(t, i):
        if type(t) != str:
            return 13
        else:
            t_split = t.split(',')
            if len(t_split) >= i + 1:
                if t_split[- (i + 1)] in dict_category:
                    return dict_category[t_split[- (i + 1)]]
                else:
                    return 13
            else:
                return 13
    for i in range(10):
        column_new = 'history{}_category'.format(i + 1)
        print('{} : read_article_ids の最新 {} 番目のカテゴリ'.format(column_new, i + 1))
        item[column_new] = item_old['read_article_ids'].map(lambda t: create_history_category(t, i))
        item[column_new] = item[column_new].astype(int)

    print('\n### pre3 done ###\n')
    print(item.columns)

    pd.options.display.max_columns = None
    pd.options.display.max_rows = None

    if mode == 'train':

        print('データサイズ:\n', item.shape)

        return item, label
    elif mode == 'validation':
        return item, label
    else:
        return item
