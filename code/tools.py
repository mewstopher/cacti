import tensorflow as tf
import os 
import tqdm
#import tqdm_notebook
import pandas as pd
import numpy as np
import cv2
import sys
sys.path.append("../../")
from cnn_utils.helper_functions import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops

fold = os.listdir("../input/train/")
train_df = pd.read_csv("../input/train.csv")

test_fold = "../input/test/"
fold = "../input/train/"
train_df.head()

def onehot():
    """
    not WORKING!!!
    """
    res = tf.one_hot(indices=[0,17499], depth=17500)
    with tf.Session() as sess:
        Y_tr= sess.run(res)
    return Y_tr.T

def one_hot(labels, C):
    """
    makes one hot from labels

    PARAMS
    --------------
    labels: label array
    C: number of classes
    """
    C = tf.constant(C, name="C")
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

def load_test_jpgs():
    """
    loads test file
    """
    X_te = []
    for i in os.listdir(test_fold):
        X_te.append(cv2.imread(test_fold + i))

    X_te = np.asarray(X_te)
    X_te = X_te.astype('float32')
    X_te /= 255
    return X_te


def split(X_tr, Y_tr):
    """
    splits train,test data
    """
    X_train, y_train, X_test, y_test = train_test_split(X_tr, Y_tr)
    return X_train, y_train, X_test, y_test


def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    creates placeholders for tensorflow variables

    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes
    """
    X = tf.placeholder(shape=(None, n_H0, n_W0, n_C0),dtype=tf.float32)
    Y = tf.placeholder(shape=(None, n_y), dtype=tf.float32)
    return X, Y


def initialize_parameters():
    """
    initializes parameters.
    shapes are the shape of filters/weights
    """

    W1 = tf.get_variable('W1', [3,3,3,64], initializer=tf.contrib.layers.xavier_initializer(seed = 0))
    W2 = tf.get_variable('W2', [3,3,64,128], initializer=tf.contrib.layers.xavier_initializer(seed = 0))
    W3 = tf.get_variable('W3', [3,3,128,256], initializer=tf.contrib.layers.xavier_initializer(seed = 0))
    W4 = tf.get_variable('W4', [3,3,256,512], initializer=tf.contrib.layers.xavier_initializer(seed = 0))
    W5 = tf.get_variable('W5', [3,3,512,512], initializer=tf.contrib.layers.xavier_initializer(seed = 0))

    ### END CODE HERE ###

    parameters = {"W1": W1,
            "W2": W2,
            "W3": W3,
            "W4": W4,
            "W5": W5
            }

    return parameters


def forward_propogation(X, parameters):
    """
    implements foward propogation

    define model here
    """
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    W4 = parameters['W4']
    W5 = parameters['W5']

    Z1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize=[1,2,2,1], strides=[1,2,2,1],
            padding='SAME')
    
    Z2 = tf.nn.conv2d(P1, W2, strides=[1,1,1,1], padding='SAME')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=[1,2,2,1], 
            strides=[1,2,2,1],padding='SAME')
    Z3 = tf.nn.conv2d(P2, W3, strides=[1,1,1,1], padding='SAME')
    A3 = tf.nn.relu(Z3)
    P3 = tf.nn.max_pool(A3, strides=[1,1,1,1],
            ksize=[1,2,2,1], padding='SAME')

    Z4 = tf.nn.conv2d(P3, W4, strides=[1,1,1,1], padding='SAME')
    A4 = tf.nn.relu(Z4)
    P4 = tf.nn.max_pool(A4, strides=[1,1,1,1],
            ksize=[1,2,2,1],padding='SAME')

    Z5 = tf.nn.conv2d(P4, W5, strides=[1,1,1,1], padding='SAME')
    A5 = tf.nn.relu(Z5)
    P5 = tf.nn.max_pool(A5, strides=[1,1,1,1],
            ksize=[1,2,2,1], padding='SAME')

    P5 = tf.contrib.layers.flatten(P5)
    Z6 = tf.contrib.layers.fully_connected(P5, 2,
            activation_fn = None)

    return Z6


def compute_cost(Z6, Y):
    """
    computes cost for output layer
    """

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z6, labels=Y))

    return cost

def model(X_train, Y_train, X_test, Y_test, test_imgs, learning_rate=.0003,
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
    n_y = Y_train.shape[1]
    costs = []

    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)

    parameters = initialize_parameters()

    Z6 = forward_propogation(X, parameters)

    cost = compute_cost(Z6, Y)

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

            print('USING ONLY FIRST MINIBATCH. DELETE AfTER TETING!')
            
            #for minibatch in minibatches:
            minibatch = minibatches[1]
            (minibatch_X, minibatch_Y) = minibatch
            _, temp_cost = sess.run([optimizer, cost],
                    feed_dict={X:minibatch_X, Y:minibatch_Y})
            minibatch_cost += temp_cost / num_minibatches

            if print_cost == True:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True:
                costs.append(minibatch_cost)


        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(Z6, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
       
       # Predict on REAL test set
        print('using test set')
        predicted_lables = np.zeros(test_imgs.shape[0])
        for i in range(0, test_imgs.shape[0]):
            predicted_lables[i*minibatch_size : (i+1)*minibatch_size] = predict_op.eval(feed_dict={X: test_imgs[i*minibatch_size : (i+1)*minibatch_size]})


        return train_accuracy, test_accuracy, parameters, predicted_lables


