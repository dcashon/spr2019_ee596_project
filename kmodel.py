from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Lambda
from keras.backend import squeeze


def build_model():
    
    model = Sequential()
    # --------------Layer 1: Conv2D + Relu + Maxpool
    model.add(Conv2D(filters=96, kernel_size=7, strides=3, input_shape=(None,
        None, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    # --------------Layer 2: Conv2D + Relu
    model.add(Conv2D(filters=256, kernel_size=3, strides=1))
    model.add(Activation('relu'))
    
    # --------------Layer 3: Conv2D + Relu
    model.add(Conv2D(filters=512, kernel_size=3, strides=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    # CLASSIFICATION HEAD
    model.add(Conv2D(filters=1024, kernel_size=19, strides=1))
    model.add(Conv2D(filters=2048, kernel_size=1, strides=1))
    model.add(Conv2D(filters=10, kernel_size=1, strides=1))
    model.add(Lambda(lambda x: x[:, 0, 0, :])) # for training!
    model.add(Activation('softmax'))
    
    return model

    
