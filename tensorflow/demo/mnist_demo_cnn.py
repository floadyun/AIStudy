from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

#加载mnist数据
dir='/home/kaka/Documents/input_data'
mnist = input_data.read_data_sets(dir, one_hot=True)
#构造计算图,使用InteractiveSession类,通过它，你可以更加灵活地构建你的代码
sess = tf.InteractiveSession()
#通过为输入图像和目标输出类别创建节点，来开始构建计算图
x = tf.placeholder('float', shape=[None, 784])
y_ = tf.placeholder('float', shape=[None, 10])
#定义权重W和偏置b
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
#通过seesion初始化变量
sess.run(tf.global_variables_initializer())

#构造softmaxfen分类函数
y = tf.nn.softmax(tf.matmul(x, w)+b)

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

for i in range(1000):
    banch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: banch[0], y_: banch[1]})
    if(i%100==0):
        print(i)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

