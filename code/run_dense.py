from dense_net import *
from tools import load_jpgs, split
X_tr, Y_tr = load_jpgs()
#Y_tr = onehot()
# print an image


index = 10 
plt.imshow(X_train[index])
plt.show()

X_train, X_test, y_train, y_test = split(X_tr, Y_tr)


#reshape X and y
X_train = X_train.reshape(X_train.shape[0], -1).T
X_test = X_test.reshape(X_test.shape[0], -1).T

#y_train = y_train.reshape(y_train.shape[0], -1).T
#y_test = y_test.reshape(y_test.shape[0], -1).T


ft = flowtools()
y_train = ft.one_hot(y_train, 2)
y_test = ft.one_hot(y_test,2)
params = ft.model(X_train, y_train, X_test, y_test, learning_rate=.0003, num_epochs=10)


