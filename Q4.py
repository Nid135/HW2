import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# download and read mnist
mnist = fetch_openml('mnist_784')

# 'mnist.data' is 70k x 784 array, each row represents the pixels from a 28x28=784 image
# 'mnist.target' is 70k x 1 array, each row represents the target class of the corresponding image
images = mnist['data']
targets = mnist['target']

# make the value of pixels from [0, 255] to [0, 1] for further process
X = mnist.data / 255.
Y = mnist.target

# split data to train and test (for faster calculation, just use 1/10 data)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X[::10], Y[::10], test_size=1000)


# TODO:use SVM with another group of parameters
from sklearn.svm import LinearSVC

X_train = X_train.reshape(Y_train.size,-1)
X_test = X_test.reshape(Y_test.size,-1)

clf=LinearSVC(penalty='l2',loss='squared_hinge',dual=True,tol=0.001,C=0.8,multi_class='ovr',fit_intercept=True,intercept_scaling=2,class_weight=None,verbose=0,random_state=None,max_iter=1000)
clf.fit(X_train,Y_train)
train_accuracy=clf.score(X_train,Y_train)
test_accuracy=clf.score(X_test,Y_test)

print('Training accuracy: %0.2f%%' % (train_accuracy*100))
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))
