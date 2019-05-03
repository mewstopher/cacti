import tensorflow as tf
import os 
import tqdm
#import tqdm_notebook
import pandas as pd
import numpy as np
import cv2
fold = os.listdir("input/train/")
train_df = pd.read_csv("input/train.csv")

fold = "input/train/"
train_df.head()

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


def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    creates placeholders for tensorflow variables

    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes
    """
    X = tf.placeholder(shape=(None, n_H0, n_W0, n_C0),dtype=float32))
    Y = tf.placeholder(shape=(None, n_y), dtype=float32)
    return X, Y


def initialize_parameters():
    """
    initializes parameters.
    shapes are the shape of filters/weights
    """

    W1 = tf.get_variable('W1', [4,4,3,8], initializer=tf.contrib.layers.xavier_initializer(seed = 0))
    W2 = tf.get_variable('W2', [2,2,8,16], initializer=tf.contrib.layers.xavier_initializer(seed = 0))
    ### END CODE HERE ###

    parameters = {"W1": W1,
            "W2": W2}

    return parameters


def forward_propogation(X, parameters):
    """
    implements foward propogation

    define model here
    """
    W1 = parameters['W1']
    W2 = parameters['W2']

    Z1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize[1,8,8,1], strides=[1,8,8,1],
            padding='SAME')
    
    Z2 = tf.nn.conv2d(P1, W2, strides=[1,1,1,1], padding='SAME')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=[1,4,4,1], 
            strides=[1,4,4,1],padding='SAME')
    P2 = tf.nn.contrib.layers.flatten(P2)
    Z3 = tf.nn.contrib.layers.fully_connected(P2, num_classes,
            activation_fn = None

    return Z3


def compute_cost(Z3, Y):
    """
    computes cost for output layer
    """

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))

    return cost

def model(X_train, Y_train, X_test, Y_test, learning_rate=.009,
        num_epochs=100, minibatch_size=64, print_cost=True):
    """
    runs model

    PARAMS
    ------
    X_train shape...
    X_test shape...
    Y_trian shape
    Y_test shape
    learning rate
    num_epochs
    minibatch_size
    """
    ops.reset_default_graph
    tf.set_random_seed(1)
    seed = 3
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape
    costs = []

    X, Y = create_placeholders(m, n_H0, n_W0, n_C0)

    parameters = initialize_parameters()

    Z3 = forward_propogation(X, parameters)

    cost = compute_cost(Z3)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            minibatch_cost =0.
            num_minibatches = int(m/minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train,
                    minibatch_size, seed)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _, temp_cost = sess.run([optimizer, cost],
                        feed_dict={X:minibatch_X, Y:minibatch_Y})
                    minibatch_cost += temp_cost / num_minibatches

            if print_cost == True and epoch % 5 == 0:
            print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
            costs.append(minibatch_cost)


    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    # Calculate the correct predictions
    predict_op = tf.argmax(Z3, 1)
    correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

    # Calculate accuracy on the test set
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(accuracy)
    train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
    test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    return train_accuracy, test_accuracy, parameters

