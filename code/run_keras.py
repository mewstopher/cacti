from keras_model  import *

X_tr, Y_tr = load_jpgs()

X_train, X_test, y_train, y_test = split(X_tr, Y_tr)

y_train = one_hot(y_train, 1).T
y_test = one_hot(y_test, 1).T

test_imgs = load_test_jpgs()

# print an image
#index = 10 
#plt.imshow(test_imgs[index])
#plt.show()

# initialize model
cactiModel = CactiModel(X_train[0].shape)

# compile model
cactiModel.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

# train model
cactiModel.fit(x = X_train, y = y_train, epochs = 2, batch_size = 32)


preds = cactiModel.evaluate(x = X_test, y =y_test )

print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

test_preds = cactiModel.predict(test_imgs)


sub_df = pd.DataFrame(test_preds, columns=['has_cactus'])

sub_df['has_cactus'] = sub_df['has_cactus'].apply(lambda x: 1 if x > 0.75 else 0)

sub_df['id'] = ''
cols = sub_df.columns.tolist()
cols = cols[-1:] + cols[:-1]
sub_df=sub_df[cols]

test_folder = os.listdir(test_fold)
for i, img in enumerate(test_folder):
        sub_df.set_value(i,'id',img)


sub_df.to_csv("../output/sub1.csv")


