#README

This is a practice problem using CNNs on a kaggle competition. The only goal here is to gain practice coding nueral networks, and a better understanding of how to effectively code them.

#Project status
---------------------------------
CNN used to identify whether or not cactus in picture.

First model (tools.py) was an attempt to code a VGG from scratch. Some thing is wrong with code, resulting in test/train accuracies of .99,1.0. 

Dense_net.py uses a fully connected 2 layer network. This one work, but accuracy is not that great.

To compare with a pretrained model, VGG16 is imported and used in run_keras.py
Results give test/train ~ .95, .90 respectively.

##TODO
1. Rename file names to appropriate things
2. Clean up pretrained code
3. code up vgg16 from scratch using both keras/tensorflow
4. find best model for competition



