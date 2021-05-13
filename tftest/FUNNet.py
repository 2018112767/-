from tensorflow.keras import layers, models, Model, Sequential


class Net_v1(Model):  # 汉字
    def __init__(self):
        super(Net_v1, self).__init__()
        self.features = Sequential([
            layers.ZeroPadding2D(padding=(1, 1)),  # 50*26*1
            layers.Conv2D(8, kernel_size=3, strides=1, activation="relu"),  # 48*24*8
            layers.MaxPool2D(pool_size=2, strides=2),  # 24*12*8
            layers.ZeroPadding2D(padding=(1, 1)),  # 26*14*8
            layers.Conv2D(16, kernel_size=3, strides=1, activation="relu"),  # 24*12*16
            layers.MaxPool2D(pool_size=2, strides=2),  # 12*6*16
            layers.ZeroPadding2D(padding=(1, 1)),  # 14*8*16
            layers.Conv2D(32, kernel_size=3, strides=1, activation="relu"),  # 12*6*32
            layers.MaxPool2D(pool_size=2, strides=2),  # 6*3*32
        ])
        self.flatten = layers.Flatten()
        self.classifier = Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(34, activation='softmax')
        ])

    def call(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


class Net_v2(Model):  # 字母
    def __init__(self):
        super(Net_v2, self).__init__()
        self.features = Sequential([
            layers.ZeroPadding2D(padding=(1, 1)),  # 50*26*1
            layers.Conv2D(8, kernel_size=3, strides=1, activation="relu"),  # 48*24*8
            layers.MaxPool2D(pool_size=2, strides=2),  # 24*12*8
            layers.ZeroPadding2D(padding=(1, 1)),  # 26*14*8
            layers.Conv2D(16, kernel_size=3, strides=1, activation="relu"),  # 24*12*16
            layers.MaxPool2D(pool_size=2, strides=2),  # 12*6*16
            layers.ZeroPadding2D(padding=(1, 1)),  # 14*8*16
            layers.Conv2D(32, kernel_size=3, strides=1, activation="relu"),  # 12*6*32
            layers.MaxPool2D(pool_size=2, strides=2),  # 6*3*32
        ])
        self.flatten = layers.Flatten()
        self.classifier = Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(35, activation='softmax')
        ])

    def call(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

"""
layers.ZeroPadding2D(padding=(1, 1)),
layers.Conv2D(8, kernel_size=3, strides=1, activation="relu"),
layers.MaxPool2D(pool_size=2, strides=2),
layers.ZeroPadding2D(padding=(1, 1)),
layers.Conv2D(8, kernel_size=3, strides=2, activation="relu"),
layers.MaxPool2D(pool_size=2, strides=2)
"""