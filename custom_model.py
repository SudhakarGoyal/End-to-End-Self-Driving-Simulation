from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation, BatchNormalization, Input, add, LSTM, TimeDistributed
from keras.models import Model

class EndtoEnd:
    def __init__(self, num_classes, input_):
        self.num_classes = num_classes
        self.input = input_

    def identity(self, input_features, filters, strides_=1):
        filter1, filter2, filter3 = filters

        x = Conv2D(filter1, kernel_size=1, strides=(1, 1))(input_features)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filter2, kernel_size=3, strides=(strides_, strides_), padding='same')(input_features)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filter3, kernel_size=1, strides=(1, 1))(input_features)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        output = add([x, input_features])


        return output

    def convolution(self, input_features, kernel, filters, strides_=1):
        x = Conv2D(filters, kernel_size=kernel, strides=(strides_, strides_))(input_features)
        x = BatchNormalization()(x)
        x = Activation('relu')(x) 

        return x

    def fully_connected(self, x):

        x = Dense(100, activation='relu')(x)
        x = Dense(50, activation='relu')(x)
        x = Dense(10, activation='relu')(x)

        x = Dense(self.num_classes)(x)

        return x

    def forward(self):
        x = Lambda(lambda x: x/127.5-1.0)(self.input)

        x = self.convolution(x, 5, 24, strides_=2)
        x = self.identity(x, [16, 16, 24], strides_=2)

        x = self.convolution(x, 5, 32, strides_=2)
        x = self.identity(x, [16, 16, 32], strides_=2)

        x = self.convolution(x, 5, 48, strides_=2)
        x = self.identity(x, [24, 24 , 48], strides_=2)

        x = self.convolution(x, 3, 64, strides_=1)
        x = self.identity(x, [32, 32, 64], strides_=1)

        x = self.convolution(x, 3, 64, strides_=1)
        x = self.identity(x, [32, 32, 64], strides_=1)
        x = Flatten()(x)

        output = self.fully_connected(x)
        model = Model(self.input, output)
        model.summary()

        return model


