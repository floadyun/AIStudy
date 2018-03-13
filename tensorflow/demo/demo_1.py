import tensorflow as tf

test1 = tf.constant([[1, 2, 3]])
test2 = tf.constant([[3, 4, 6]])

sess = tf.Session()
result = sess.run(test1*test2)
sess.close()
print(result)
