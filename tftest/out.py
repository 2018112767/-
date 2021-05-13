import numpy as np
from FUNNet import Net_v1, Net_v2
from keras.utils import plot_model
import pydot
import os

model1 = Net_v1()  # 汉字
#model1.load_weights('./save_weights/Chinese.ckpt')

model2 = Net_v2()  # 字母
#model2.load_weights('./save_weights/Chars.ckpt')
np.set_printoptions(threshold=np.inf)

for lay in model1.layers:
    with open('Zhwgt.txt', 'a+', encoding='utf-8') as f:
        weights = lay.get_weights()
        print(np.array(weights), file=f)
        f.close()

for lay in model2.layers:
    with open('Chwgt.txt', 'a+', encoding='utf-8') as f:
        weights = lay.get_weights()
        print(np.array(weights), file=f)
        f.close()