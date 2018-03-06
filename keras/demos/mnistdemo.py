from keras import models
from keras.preprocessing import image
import numpy as np
from keras.applications.resnet50 import preprocess_input,decode_predictions


model = models.load_model('number\mnist_cnn.h5')

img = image.load_img('number\\number_9.jpg', target_size=(28, 28))
grayImage = img.convert('L')   #转化为灰度图
#图片预处理
x = image.img_to_array(grayImage)
x_test = np.expand_dims(x, axis=0)
# 插入这一个轴是关键，因为keras中的model的tensor的shape是（bath_size, h, w, c),如果是tf后台
x_test = preprocess_input(x_test, mode='tf')

# 对输入进行预测
preds = model.predict(x_test)
classes = model.predict_classes(x_test)

# 输出预测概率
print('预测概率:', preds)
print('预测值:', classes[0])
