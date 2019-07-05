import tensorflow as tf
a = tf.constant([[1,2,3],[1,2,3]])

b = tf.constant([[2,3,4]])

b_1 = tf.constant([[2,3],[1,2],[3,4]])
print("a",a)
print("b",b)
print("b_1",b_1)

# c = a*b # (2,3)*(1,3)->(2,3)　两个矩阵中对应元素各自相乘
# print("c",c)

d = tf.multiply(a,b)# (2,3)*(1,3)->(2,3)　两个矩阵中对应元素各自相乘
print("d",d)

# e = tf.matmul(a,b_1)# 矩阵乘法(2,3)*(3,2)->(2,2)
# print("e",e)
# c = tf.Print(c,[c])

with tf.Session() as sess:
    # print(sess.run(c))
    print(sess.run(d))
    # print(sess.run(e))
