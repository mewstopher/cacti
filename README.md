# README
Babas First CNN

This is a practice problem using CNNs on a kaggle competition. The only goal here is to gain practice coding nueral networks, and a better understanding of how to effectively code them.

# Project status
---------------------------------
CNN used to identify whether or not cactus in picture.

First model (tools.py) was an attempt to code a VGG from scratch. Some thing is wrong with code, resulting in test/train accuracies of .99,1.0. 

Dense_net.py uses a fully connected 2 layer network. This one work, but accuracy is not that great.

To compare with a pretrained model, VGG16 is imported and used in run_keras.py
Results give test/train ~ .95, .90 respectively.

Training model (VGG) from scrach works, but pretained model is (obviously) a much better option for accuracy. 

Training last 2 layers of pretrained VGG results in .98 accuracy on actual test data. While this is not great for the competiton results, it is good enough to move on. 

List of files:
1. keras_model: codes up a VGG from scratch. 
2. Dense_net: a fully connected 2 layer neural net
3. imports: contains all imported modules used in model scripts
4. run_dense/keras: script for running dense and VGG(from scratch)
5. run VGG: runs pre-trained VGG
6. tf_VGG: VGG16 coded from scatch using tensorflow

