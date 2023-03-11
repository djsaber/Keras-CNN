#coding=gbk

from keras.layers import Conv2D, Dense, MaxPooling2D, Input, Flatten
from keras.models import Model


class Simple_CNN(Model):
    def __init__(self, input_shape, output_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = Conv2D(
            filters=32,
            kernel_size=3,
            padding="same",
            activation="relu"
            )
        self.conv2 = Conv2D(
            filters=64,
            kernel_size=3,
            padding="same",
            activation="relu"
            )
        self.conv3 = Conv2D(
            filters=128,
            kernel_size=3,
            padding="same",
            activation="relu"
            )  
        self.maxp = MaxPooling2D()
        self.flatten = Flatten()
        self.dense = Dense(output_dim, activation="softmax")

        self.input_layer = Input(input_shape)
        self.out = self.call(self.input_layer)
    

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.maxp(x)
        x = self.conv2(x)
        x = self.maxp(x)
        x = self.conv3(x)
        x = self.maxp(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x