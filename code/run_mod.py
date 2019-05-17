from tools  import *

X_tr, Y_tr = load_jpgs()

X_train, X_test, y_train, y_test = split(X_tr, Y_tr)

y_train = one_hot(y_train, 2).T
y_test = one_hot(y_test, 2).T

test_imgs = load_test_jpgs()

# print an image
index = 10
plt.imshow(X_train[index])
plt.show()



_, _, params, preds = model(X_train, y_train, X_test, y_test, test_imgs, learning_rate=.0003, num_epochs=1)


xp=[]
for j  in dfp.id.values:
    for i in os.listdir(fold):
        if i == j:
            xp.append(i)




