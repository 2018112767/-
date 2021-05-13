from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from FUNNet import Net_v1, Net_v2
import tensorflow as tf
import cv2
import os
import numpy as np
import torch

CHARS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', '0', '1', '2', '3', '4',
         '5', '6', '7', '8', '9', 'O'
         ]

ZHARS = ['京', '津', '沪', '渝', '冀', '豫', '云', '辽', '黑', '湘',
         '皖', '鲁', '新', '苏', '浙', '赣', '鄂', '桂', '甘', '晋',
         '蒙', '陕', '吉', '闽', '贵', '粤', '青', '藏', '川', '宁',
         '琼', '港', '澳', '台'
         ]

dict = {}
threshold = 127.5

def init():
    for i in range (0, 34):
        dict[ZHARS[i]] = i
    for i in range (0, 35):
        dict[CHARS[i]] = i
        dict['O'] = 0
    if not os.path.exists("try_weights"):
        os.makedirs("try_weights")


#读取字母
def data_utilschars(path):
    chars_train = []
    chars_label = []

    for root, dirs, files in os.walk(path):
        for filename in files:
            root_int = filename
            kw = os.path.join(root, filename)
            Image = cv2.imdecode(np.fromfile(kw, dtype=np.uint8), 0)
            for i in range(1, 7):
                chars_label.append(root_int[i])
                if i == 1:
                    box = (93, 30, 165, 160)
                else:
                    box = (200+(i-2)*72, 30, 200+(i-1)*72, 160)
                digit_img = Image[box[1]:box[3], box[0]:box[2]]
                digit_img = cv2.resize(digit_img, (48, 24))
                digit_img[digit_img > threshold] = 255.0  # 二值化
                digit_img[digit_img <= threshold] = 0
                #digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
                chars_train.append(digit_img)
                #cv2.imshow(root_int[i], digit_img)
                #cv2.waitKey(100)

    for i in range(0, len(chars_label)):
        chars_label[i] = dict[chars_label[i]]

    train_set = chars_train
    train_lable = chars_label
    return np.array(train_set).astype(float), np.array(train_lable)


#读取汉字
def data_utilscharsChinese(path):
    chars_train = []
    chars_label = []

    for root, dirs, files in os.walk(path):
        for filename in files:
            root_int = filename
            path = os.path.join(root, filename)
            Image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 0)
            chars_label.append(root_int[0])
            box = (20, 30, 93, 160)
            digit_img = Image[box[1]:box[3], box[0]:box[2]]
            digit_img = cv2.resize(digit_img, (48, 24))
            digit_img[digit_img > threshold] = 255.0  # 二值化
            digit_img[digit_img <= threshold] = 0
            # digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
            chars_train.append(digit_img)
            #cv2.imshow(root_int[0], digit_img)
            #cv2.waitKey(100)

    for i in range(0, len(chars_label)):
        chars_label[i] = dict[chars_label[i]]

    train_set = chars_train
    train_lable = chars_label
    return np.array(train_set).astype(float), np.array(train_lable)


#训练
def train_mlp(flag):
    # GPU运算
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 识别数字
    if flag == 2:  # 字母
        model = Net_v2()
        train_set, train_lable, = data_utilschars("./Atrain")
        test_set, test_lable, = data_utilschars("./test")
    # 识别汉字
    else:  # 汉字
        model = Net_v1()
        train_set, train_lable, = data_utilscharsChinese("./Atrain")
        test_set, test_lable, = data_utilscharsChinese("./test")
    # 先训练数字和字母#################################################################################
    train_set = train_set / 255.0
    train_set = train_set[..., tf.newaxis]

    # create data generator
    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_set, train_lable)).shuffle(155).batch(5)

    test_set = test_set / 255.0
    test_set = test_set[..., tf.newaxis]
    test_ds = tf.data.Dataset.from_tensor_slices(
        (test_set, test_lable)).shuffle(65).batch(5)

    # define loss
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    # define optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):
        predictions = model(images)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

    np.set_printoptions(threshold=np.inf)
    EPOCHS = 200  # 迭代次数
    best_test_loss = float('inf')
    for epoch in range(EPOCHS):
        train_loss.reset_states()        # clear history info
        train_accuracy.reset_states()    # clear history info
        test_loss.reset_states()         # clear history info
        test_accuracy.reset_states()     # clear history info

        for images, labels in train_ds:
            train_step(images, labels)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))
        if test_loss.result() < best_test_loss:
            best_test_loss < test_loss.result()  #更新模型
            if flag == 2:
                model.save_weights("./try_weights/Chars.ckpt", save_format='tf')
            else:
                model.save_weights("./try_weights/Chinese.ckpt", save_format='tf')

    if flag == 1:
        for lay in model.layers:
            with open('Zhwgt.txt', 'a+', encoding='utf-8') as f:
                weights = lay.get_weights()
                print(np.array(weights), file=f)
                f.close()
    else:
        for lay in model.layers:
            with open('Chwgt.txt', 'a+', encoding='utf-8') as f:
                weights = lay.get_weights()
                print(np.array(weights), file=f)
                f.close()


if __name__ == "__main__":
    init()
    train_mlp(1)  # 汉字
    train_mlp(2)  # 字母
