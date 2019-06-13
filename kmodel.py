from keras.models import Sequential, load_model, Model
from keras.layers import Conv2D, MaxPooling2D, Activation, Lambda, Input, Dropout, Softmax
from keras.backend import squeeze


def build_model_classifier():
    """
    Original model. Found to have some issues
    """
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

def build_model_regressor(model_path):
    """
    The bounding box regressor is trained using the same feature map from the classifier.
    Loads in the classification model, but freezes the feature map layers
    """
    # load the best classification model
    classifier = load_model(model_path)
    # freeze layers
    for l in classifier.layers:
        l.trainable = False
    # construct the regressor model on top
    feature_mapper = Model(inputs=classifier.input, outputs=classifier.layers[-6].output)
    x = Input(shape=(None, None, 3))
    regressor = feature_mapper(x)
    regressor = Conv2D(filters=4096, kernel_size=3, strides=1)(regressor)
    regressor = Conv2D(filters=1024, kernel_size=1, strides=1)(regressor)
    regressor = Conv2D(filters=4, kernel_size=1, strides=1)(regressor)
    new_model = Model(inputs=x, outputs=regressor)
    return new_model
    
    
def build_model_accurate():
    """
    A deeper model, found to be more accurate 
    """
    model = Sequential()
    
    # --------------Layer 1: Conv2D + Relu + Maxpool
    model.add(Conv2D(filters=96, kernel_size=10, strides=5, input_shape=(None,
        None, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    # --------------OUTPUT 25*25
    
    # --------------Layer 2: Conv2D + Relu + Maxpool
    model.add(Conv2D(filters=256, kernel_size=4, strides=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    #---------------OUTPUT 11*11
    
    # --------------Layer 3: Conv2D + Relu
    model.add(Conv2D(filters=512, kernel_size=4, strides=1))
    model.add(Activation('relu'))
    #---------------OUTPUT 8*8

    #---------------Layer 4: Conv2D + Relu + Maxpool
    model.add(Conv2D(filters=1024, kernel_size=3, strides=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    #--------------OUTPUT 3*3

    # CLASSIFICATION HEAD
    model.add(Conv2D(filters=3072, kernel_size=3, strides=1))
    model.add(Conv2D(filters=4096, kernel_size=1, strides=1))
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=10, kernel_size=1, strides=1))
    model.add(Lambda(lambda x: x[:, 0, 0, :])) # for training!
    model.add(Softmax(axis=3))
    
    return model
    

    
