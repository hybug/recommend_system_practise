# -*- coding: utf-8 -*-
# @Time    : 2019/9/4 14:42
# @Author  : hanyu
# @mail    ：hanyu01@mail.jj.cn

import pandas as pd
import os

def load_data():
    train_data = {}

    file_path = './data/tiny_train_input.csv'
    data = pd.read_csv(file_path, header=None)
    data.columns = ['col' + str(i) for i in range(data.shape[1])]
    label = data.col0.values
    label = label.reshape(len(label), 1)
    train_data['train_y'] = label

    continuous_feature = pd.DataFrame()
    category_feature = pd.DataFrame()
    continuous_feautre_name = []
    category_feature_name = []
    # 构建feature-index对，其中离散变量有多少个离散值分为多少类
    # feature_dict = {'col1':1, 'col2':2, 'col3':{0:3, 1:4}}
    feature_dict = {}
    cnt = 1
    for index in range(1, data.shape[1]):
        feature = data.iloc[:, index]
        col = feature.name
        # 判断哪些是离散特征，哪些是连续特征
        feature_length = len(set(feature))

        if feature_length > 10: # 连续特征
            # 归一化
            feature = (feature - feature.mean()) / feature.std()
            continuous_feature = pd.concat([continuous_feature, feature], axis=1)
            feature_dict[col] = cnt
            cnt += 1
            continuous_feautre_name.append(col)
        else: # 离散特征
            # 取出离散值
            category_set = feature.unique()
            print(category_set)
            # 有多少个离散值，one-hot成多少个特征
            # 比如 gender: 0, 1, 2 ===> gender-0, gender-1, gender-2
            feature_dict[col] = dict(zip(category_set, range(cnt, cnt + len(category_set))))
            category_feature = pd.concat([category_feature, feature], axis=1)
            cnt += len(category_set)
            category_feature_name.append(col)

    feature_dim = cnt
    # feature_value = [连续值,连续值,连续值,...,1.0,1.0]
    feature_value = pd.concat([continuous_feature, category_feature], axis=1)
    # feature_index = [特征index,特征index,特征index,。。。]
    # 特征index指feature_dict中的value值，而不是col0这样的col名
    feature_index = feature_value.copy()

    for i in feature_index.columns:
        if i in continuous_feautre_name: # 如果是连续特征，可以直接按照dict中的赋值
            feature_index[i] = feature_dict[i]
        else:
            feature_index[i] = feature_index[i].map(feature_dict[i])
            feature_value[i] = 1.0

    train_data['x_index'] = feature_index.values.tolist()
    train_data['x_value'] = feature_value.values.tolist()
    train_data['feat_dim'] = feature_dim
    return train_data






if __name__ == '__main__':
    load_data()