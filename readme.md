dataset from kaggle

$ pip install opencv-contrib-python
$ pip install numpy
$ pip install scikit-learn
$ pip install scikit-image
$ pip install imutils
$ pip install matplotlib
$ pip install tensorflow

# TrafficSignNet

Conv2D
input: (None, 32, 32, 3)
output: (None, 32, 32, 8)

Activation
input: (None, 32,32,8)
output: (None, 32,32,8)

BatchNormalization
input: (None, 32, 32, 8)
output: (None, 32, 32, 8)

MaxPooling2D
input: (None, 32, 32, 8)
output: (None, 16, 16, 8)


-----
`relu` => Rectified Linear Unit

Batch Normalization is used to normalize the activations of a given input volume before passing it to the next layer in the network.  
It has been proven to be very effective at reducing the number of epochs required to train a CNN as well as stabilizing training itself.

MaxPooling2D: progressively reducing the spatial size

Dropout : n% of the node connections are randomly disconnected (dropped out) between layers during each training iteration. concept not to be overlooked


A big thanks to Thomas Tracey who proposed using CLAHE to improve traffic sign recognition [in his 2017 article](https://medium.com/@thomastracey/recognizing-traffic-signs-with-cnns-23a4ac66f7a7).

///problem 
having trouble with class weigth 
lets https://datascience.stackexchange.com/questions/13490/how-to-set-class-weights-for-imbalanced-classes-in-keras

Let's import the module first

from sklearn.utils import class_weight

In order to calculate the class weight do the following

class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)
Thirdly and lastly add it to the model fitting

model.fit(X_train, y_train, class_weight=class_weights)

