from tools import *

X_tr, Y_tr = load_jpgs()
Y_tr = onehot()
X_train, X_test, y_train, y_test = split(X_tr, Y_tr)

_, _, params = model(X_train, y_train, X_test, y_test, learning_rate=.009, num_epochs=3)


