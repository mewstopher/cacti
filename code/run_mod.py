from tools  import *

X_tr, Y_tr = load_jpgs()

X_train, X_test, y_train, y_test = split(X_tr, Y_tr)

y_train = one_hot(y_train, 2).T
y_test = one_hot(y_test, 2).T

# print an image
index = 10 
plt.imshow(X_train[index])
plt.show()



_, _, params = model(X_train, y_train, X_test, y_test, learning_rate=.009, num_epochs=3)


