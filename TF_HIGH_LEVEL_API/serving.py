# -*- coding: utf-8 -*-
# @Time    : 2019/8/9 15:50
# @Author  : hanyu
# @mail    ：hanyu01@mail.jj.cn


import tensorflow as tf
import pandas as pd

# 从导出目录中加载模型，并生成预测函数。
predict_fn = tf.contrib.predictor.from_saved_model('./savedmodel/model2/')

# 使用 Pandas 数据框定义测试数据。
inputs = pd.DataFrame({
    'SepalLength': [5.1, 5.9, 6.9],
    'SepalWidth': [3.3, 3.0, 3.1],
    'PetalLength': [1.7, 4.2, 5.4],
    'PetalWidth': [0.5, 1.5, 2.1],
})

# 将输入数据转换成序列化后的 Example 字符串。
examples = []
for index, row in inputs.iterrows():
    feature = {}
    for col, value in row.iteritems():
        feature[col] = tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    examples.append(example.SerializeToString())

# 使用tensor作为inputs
# inputs = {
#     'SepalLength': tf.convert_to_tensor([5.1, 5.9, 6.9]),
#     'SepalWidth': tf.convert_to_tensor([3.3, 3.0, 3.1]),
#     'PetalLength': tf.convert_to_tensor([1.7, 4.2, 5.4]),
#     'PetalWidth': tf.convert_to_tensor([0.5, 1.5, 2.1]),
# }
inputs = {
    'SepalLength': [5.1, 5.9, 6.9],
    'SepalWidth': [3.3, 3.0, 3.1],
    'PetalLength': [1.7, 4.2, 5.4],
    'PetalWidth': [0.5, 1.5, 2.1],
}




# 开始预测
import time
time1 = time.time()
# predictions = predict_fn({'examples': examples})
predictions = predict_fn(inputs)
print(time.time() - time1) # 0.043025970458984375
print(predictions)


# {
#     'classes': [
#         [b'0', b'1', b'2'],
#         [b'0', b'1', b'2'],
#         [b'0', b'1', b'2']
#     ],
#     'scores': [
#         [9.9826765e-01, 1.7323202e-03, 4.7271198e-15],
#         [2.1470961e-04, 9.9776912e-01, 2.0161823e-03],
#         [4.2676111e-06, 4.8709501e-02, 9.5128632e-01]
#     ]
# }