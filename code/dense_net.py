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

fold = "../input/train/"
train_df.head()



#from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

# read in data
# train data has 42000 rows, 785 variabes
# each picture should be 28*28
# first column is the label

class flowtools():
    
    def placeholder(self, n_x, n_y):
        """
        creates placeholder for tf session
        
        PARAMS
        -----------
        n_x: size of image (28x28)
        n_y: number of classes

        """
        X = tf.placeholder(tf.float32, [n_x, None])
        Y = tf.placeholder(tf.float32, [n_y, None])
        return(X, Y)

    def one_hot(self,labels, C):
        """
        makes one hot matrx from labels 

        PARAMS
        /-----------
        labels: label array
        C: number of classes
        """
        C = tf.constant(C, name ="C")
        one_hot_mat = tf.one_hot(labels, C, axis=0)
        with tf.Session() as sess:
            one_hot = sess.run(one_hot_mat)
        return(one_hot)

    def init_params(self, X_train):
        """
        initialize parameters randomly

        PARAMS
        ------------
        need to add params to make more flexible
        """
        W1 = tf.get_variable("W1", [500, X_train.shape[0]], 
            initializer=tf.contrib.layers.xavier_initializer(seed=1))
        b1 = tf.get_variable("b1", [500, 1], initializer=tf.zeros_initializer()) 
        W2 = tf.get_variable("W2", [50, 500], 
                initializer=tf.contrib.layers.xavier_initializer(seed=1))
        b2 = tf.get_variable("b2", [50, 1], 
                initializer=tf.zeros_initializer())
        W3 = tf.get_variable("W3", [2, 50],
                initializer=tf.contrib.layers.xavier_initializer(seed=1))
        b3 = tf.get_variable("b3", [2, 1], initializer=tf.zeros_initializer())

        parameters = {"W1" : W1,
                      "b1" : b1,
                      "W2" : W2,
                      "b2" : b2,
                      "W3" : W3,
                      "b3" : b3
                      }
        return(parameters)


    def forward_prop(self,X, parameters):
        """
        conducts forward propogation
        linear:relu: linear:relu: linear:softmax

        PARAMS
        -----------
        X: input dataset placeholder of with variables as rows,
            obs as columns
        parameters: initialized parameters (dictionary)
        """
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        W3 = parameters["W3"]
        b3 = parameters["b3"]

        Z1 = tf.add(tf.matmul(W1, X), b1)
        A1 = tf.nn.relu(Z1)
        Z2 = tf.add(tf.matmul(W2, A1), b2)
        A2 = tf.nn.relu(Z2)
        Z3 = tf.add(tf.matmul(W3, A2), b3)
        
        return(Z3)


    def comp_cost(self,Z3, Y):
        """
        computes cost using last layer and label(Y)

        PARAMS
        ------------
        Z3: output of last layer forward prop
        Y: label of Z3, same shape as Z3
        """

        logits = tf.transpose(Z3)
        labels = tf.transpose(Y)

        cost =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

        return(cost)

    
    def model(self,X_train,  y_train, X_test, y_test, learning_rate=.0001,
            num_epochs=2, minibatch_size=32, print_cost=True):
        """
        runs model

        PARAMS
        ------------

        """
        ops.reset_default_graph()
        tf.set_random_seed(1)
        seed = 3
        (n_x, m) = X_train.shape
        n_y = y_train.shape[0]
        costs = []

        X, Y = self.placeholder(n_x, n_y)
        parameters= self.init_params(X_train)
        Z3  = self.forward_prop(X, parameters)
        cost = self.comp_cost(Z3, Y)
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
        init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(num_epochs):
                epoch_cost = 0.
                num_minibatches = int(m / minibatch_size)
                seed = seed + 1
                minibatches = random_mini_batches_flat(X_train, y_train, minibatch_size, seed)
                for minibatch in minibatches:
                    (minibatch_X, minibatch_Y) = minibatch
                    _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})
                    epoch_cost += minibatch_cost / num_minibatches

                if print_cost == True:
                    print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
                if print_cost == True:
                    costs.append(epoch_cost)
                
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()
            
            parameters = sess.run(parameters)
            print ("Parameters have been trained!")
            correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print ("Train Accuracy:", accuracy.eval({X: X_train, Y: y_train}))
            print ("Test Accuracy:", accuracy.eval({X: X_test, Y: y_test}))

            return(parameters)



        


