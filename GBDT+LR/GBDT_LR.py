import lightgbm as lgb

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression

print('Loading data...')
df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')

numeric_col = ["ps_reg_01", "ps_reg_02", "ps_reg_03", "ps_car_12", "ps_car_13", "ps_car_14", "ps_car_15"]

y_train = df_train['target']
# y_test = df_test['target']
x_train = df_train[numeric_col]
x_test = df_train[numeric_col]

lgb_train = lgb.Dataset(x_train, y_train)
# lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 64,
    'num_trees': 100,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

num_leaf = 64

print('Start training...')
gbm = lgb.train(params, lgb_train, num_boost_round=100)
print('Save model...')
gbm.save_model('model.txt')

print('Start predicting...')
# get the output index from gbdt
y_pred = gbm.predict(x_train, pred_leaf=True)

print(np.array(y_pred).shape)
print(y_pred[:10])

print('Writing transformed training data')
# get one-hot metrix
transformed_training_matrix = np.zeros([len(y_pred), len(y_pred[0]) * num_leaf], dtype=np.int64)
for i in range(len(y_pred)):
    temp = np.arange(len(y_pred[0])) * num_leaf + np.array(y_pred[i])
    transformed_training_matrix[i][temp] += 1

lr = LogisticRegression(penalty='l2', C=0.05)
lr.fit(transformed_training_matrix, y_train)

y_pred_lr = lr.predict_proba(transformed_training_matrix)

print(y_pred_lr)