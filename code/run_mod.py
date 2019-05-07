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



#TESTING

tf.reset_default_graph()

with tf.Session() as sess:
    np.random.seed(1)
    X, Y = create_placeholders(32, 32, 3,2 )
    parameters = initialize_parameters()
    Z6 = forward_propogation(X, parameters)
    init = tf.global_variables_initializer()
    sess.run(init)
    a = sess.run(Z6, {X: np.random.randn(2,32,32,3), Y: np.random.randn(2,2)})
    print("Z6 = " + str(a))


tf.reset_default_graph()

with tf.Session() as sess:
    np.random.seed(1)
    X, Y = create_placeholders(32, 32, 3, 2)
    parameters = initialize_parameters()
    Z6 = forward_propogation(X, parameters)
    cost = compute_cost(Z6, Y)
    init = tf.global_variables_initializer()
    sess.run(init)
    a = sess.run(cost, {X: np.random.randn(4,32,32,3), Y: np.random.randn(4,2)})
    print("cost = " + str(a))




