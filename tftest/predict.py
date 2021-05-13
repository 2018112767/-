import os
import Ntrain
import numpy as np
import matplotlib.pyplot as plt
import cv2
from FUNNet import Net_v1, Net_v2

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
threshold = 127.5


def main(path):
    im_height = 48
    im_width = 24

    model1 = Net_v1()  # 汉字
    model1.load_weights('./try_weights/Chinese.ckpt')

    model2 = Net_v2()  # 字母
    model2.load_weights('./try_weights/Chars.ckpt')

    a = 0  # 成功识别数量
    for root, dirs, files in os.walk(path):
        for filename in files:
            str = ""
            root_int = filename
            kw = os.path.join(root, filename)
            Image = cv2.imdecode(np.fromfile(kw, dtype=np.uint8), 0)

            box = (20, 30, 93, 160)
            digit_img = Image[box[1]:box[3], box[0]:box[2]]
            #cv2.imshow(root_int[0], digit_img)
            #cv2.waitKey(100)
            digit_img = cv2.resize(digit_img, (48, 24))
            digit_img[digit_img > threshold] = 255.0  # 二值化
            digit_img[digit_img <= threshold] = 0
            digit_img = np.array(digit_img).astype(float)
            digit_img = digit_img / 255.
            digit_img = (np.expand_dims(digit_img, 0))
            digit_img = (np.expand_dims(digit_img, 3))
            result = np.squeeze(model1.predict(digit_img))
            predict_class = np.argmax(result)
            str = str + ZHARS[predict_class]

            for i in range(1, 7):
                if i == 1:
                    box = (93, 30, 165, 160)
                else:
                    box = (200+(i-2)*72, 30, 200+(i-1)*72, 160)
                digit_img = Image[box[1]:box[3], box[0]:box[2]]
                #cv2.imshow(root_int[i], digit_img)
                #cv2.waitKey(100)
                digit_img = cv2.resize(digit_img, (48, 24))
                digit_img[digit_img > threshold] = 255.0  # 二值化
                digit_img[digit_img <= threshold] = 0
                digit_img = np.array(digit_img).astype(float)
                digit_img = digit_img / 255.
                digit_img = (np.expand_dims(digit_img, 0))
                digit_img = (np.expand_dims(digit_img, 3))
                result = np.squeeze(model2.predict(digit_img))
                predict_class = np.argmax(result)
                str = str + CHARS[predict_class]
            str = str + '.BMP'
            if str == filename:  #识别正确
                a = a + 1
            else:  # 错误
                print(filename)
                print(str)
                print("")
        print(a)


if __name__ == '__main__':
    main('./ktest')