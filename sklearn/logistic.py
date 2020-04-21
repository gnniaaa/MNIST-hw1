#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
mnist= fetch_openml("mnist_784")
images = mnist.data
targets = mnist.target
X = mnist.data / 255.
Y = mnist.target
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X[::10], Y[::10], test_size=1000)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
clf = LogisticRegression()
clf.fit(X_train,Y_train)
Y_train1=clf.predict(X_train)
Y_test1=clf.predict(X_test)
sum1=0
sum2=0
for index in range(len(Y_train1)):
    if Y_train1[index] == Y_train[index]:
        sum1=sum1+1
for index in range(len(Y_test1)):
    if Y_test1[index] == Y_test[index]:
        sum2=sum2+1
train_accuracy=sum1/len(Y_train1)
test_accuracy=sum2/len(Y_test1)
print('Training accuracy: %0.2f%%' % (train_accuracy*100))
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))