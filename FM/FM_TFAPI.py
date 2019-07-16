import numpy as np


def vectorize_dic(feature_dict, user_itemsnum=None, num_col_matrix=None, n=0, num_feature=0):
    """
    :param feature_dict: {'users':array, 'items':array}
    :param user_itemsnum: {'1user':262, '2user':52, ...}
    :param num_col_matrix: len(user_itemnum)
    :param n: number of sample
    :param num_feature: number of feature
    :return:
    """
    if user_itemsnum == None:
        user_itemsnum = dict()

    # 安装user-item-user-item顺序提供index参考
    # [1,1,2,1,3,1]---对应user-item矩阵的坐标
    col_index = np