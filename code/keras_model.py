from imports import *
fold = "../input/train/"
test_fold = "../input/test/"





def CactiModel(input_shape):
    """
    Implementation of the HappyModel.

    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """
    X_input = Input(input_shape)

    X = Conv2D(64, (3,3), strides=(1,1),padding='same',name = 'Conv0')(X)
    X = Activation('relu')(X)

    X = Conv2D(64, (3,3), strides=(1,1),padding='same',name = 'Conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2), strides=(2,2),name='max_pool0')(X)

    X = Conv2D(128, (3,3), strides=(1,1),padding='same',name = 'Conv4')(X)
    X = Activation('relu')(X)
    X = Conv2D(128, (3,3), strides=(1,1),padding='same',name = 'Conv5')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2), strides=(2,2), name='max_pool1')(X)

    X = Conv2D(256, (3,3), strides=(1,1),padding='same',name = 'Conv6')(X)
    X = Activation('relu')(X)
    X = Conv2D(256, (3,3), strides=(1,1),padding='same',name = 'Conv7')(X)
    X = Activation('relu')(X)
    X = Conv2D(256, (3,3), strides=(1,1),padding='same',name = 'Conv8')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2), strides=(2,2), name='max_pool2')(X)


    X = Conv2D(512, (3,3), strides=(1,1),padding='same',name = 'Conv9')(X)
    X = Activation('relu')(X)
    X = Conv2D(512, (3,3), strides=(1,1),padding='same',name = 'Convd10')(X)
    X = Activation('relu')(X)
    X = Conv2D(512, (3,3), strides=(1,1),padding='same',name = 'Conv11')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2), strides=(2,2), name='max_pool2')(X)

    X = Conv2D(512, (3,3), strides=(1,1),padding='same',name = 'Conv12')(X)
    X = Activation('relu')(X)
    X = Conv2D(512, (3,3), strides=(1,1),padding='same',name = 'Convd13')(X)
    X = Activation('relu')(X)
    X = Conv2D(512, (3,3), strides=(1,1),padding='same',name = 'Conv14')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2), strides=(2,2), name='max_pool2')(X)

    X = Flatten()(X)
    X = Dense(4096, activation='relu',name='fc1')(X)
    X = Dense(4096, activation='relu',name='fc2')(X)
    X = Dense(1, activation='sigmoid', name='fc3')(X)

    model = Model(inputs = X_input, outputs=X, name= 'cactiVGG')

    return model



