### This is a simple example to showcase the use of this class, this example give an approximate for the relation y=2*x
import numpy as np
import Linear_regression_class as lrc
alpha=0.01 # learning rate
x=np.array([[1],[2],[3],[4]])
y=np.array([2,4,6,8])
b=0.0
n=y.shape[0] # number of training elements
m=x.shape[1] # number of properties
w=np.zeros((1,m))

test=lrc.linear_regression(x,y,w,b,alpha)
test.train(120)
testcase=np.array([0])
print(test.f(testcase))

