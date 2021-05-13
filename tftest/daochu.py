import tensorflow as tf
import h5py
from FUNNet import Net_v1, Net_v2
import numpy as np

model1 = Net_v1()  # 汉字
model1.load_weights('./save_weights/Chinese.ckpt').expect_partial()

model2 = Net_v2()  # 字母
model2.load_weights('./save_weights/Chars.ckpt').expect_partial()
np.set_printoptions(threshold=np.inf)

for lay in model1.layers:
        weights = lay.get_weights()
        for i in range(len(weights)):
            f = open('./save_weights' + '/weight_layer_' + format(i, '01d') + '.txt', 'w')

for lay in model2.layers:
    with open('Chwgt.txt', 'a+', encoding='utf-8') as f:
        f.write(lay.name + '\n')
        weights = lay.get_weights()
        for i in range(len(weights)):
            f.write('shape:' + str(weights[i].shape) + '\n')
            f.write(str(weights[i]) + '\n')