import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
from logisticClassify2 import *

iris = np.genfromtxt("data/iris.txt", delimiter=None)
X, Y = iris[:,0:2], iris[:,-1] # get first two features & target 3
X,Y = ml.shuffleData(X,Y) # reorder randomly (important later)
X,_ = ml.transforms.rescale(X) # works much better on rescaled data

# 1.
XA, YA = X[Y < 2, :], Y[Y < 2]
XB, YB = X[Y>0,:], Y[Y>0]

XA0, YA0 = XA[YA ==0, :], Y[Y ==0]
XA1, YA1 = XA[YA ==1, :], Y[Y ==1]

XB0, YB0 = XB[YB ==1, :], Y[Y ==1]
XB1, YB1 = XB[YB ==2, :], Y[Y ==2]

# plt.scatter(XB0[:,0],XB0[:,1], c = 'r')
# plt.scatter(XB1[:,0],XB1[:,1], c = 'b')
# plt.xlabel("parameter 1")
# plt.ylabel("parameter 2")
# plt.title("class 1 vs 2")
#
#
# plt.show()

learner = logisticClassify2();
learner.classes = np.unique(YA)
wts = np.array([0.5,0.25,1])
learner.theta = wts;

print()