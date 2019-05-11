from imports import *
fold = "../input/train/"
test_fold = "../input/test"



def onehot():
    res = tf.one_hot(indices=[0,17499], depth=17500)
    with tf.Session() as sess:
        Y_tr= sess.run(res)
    return Y_tr.T


def one_hot(labels, C):
    """
    makes one hot from labels

    PARAMS
    ------
    labels: label array
    C: number of classes
    """
    C = tf.constant(C, name='C')
    one_hot_mat = tf.one_hot(labels, C, axis=0)
    with tf.Session() as sess:
        one_hot = sess.run(one_hot_mat)

    return one_hot


def load_jpgs():
    """
    loads jpgs, labels into X_tr, Y_tr
    """
    X_tr = []
    Y_tr = []
    imges = train_df['id'].values
    for img_id in imges:
        X_tr.append(cv2.imread(fold + img_id))    
        Y_tr.append(train_df[train_df['id'] == img_id]['has_cactus'].values[0])  

    X_tr = np.asarray(X_tr)
    X_tr = X_tr.astype('float32')
    X_tr /= 255
    Y_tr = np.asarray(Y_tr)

    return X_tr, Y_tr


def split(X_tr, Y_tr):
    """
    splits train,test data
    """
    X_train, y_train, X_test, y_test = train_test_split(X_tr, Y_tr)
    return X_train, y_train, X_test, y_test



def CactiModel(input_shape):
    """
    Implementation of the HappyModel.

    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """
    X_input = Input(input_shape)

    X = ZeroPadding2D((3,3))(X_input)

    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((2, 2), name='max_pool')(X)

    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    model = Model(inputs = X_input, outputs = X, name='HappyModel')
     
    return model



