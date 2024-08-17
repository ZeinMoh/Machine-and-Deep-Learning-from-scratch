###This example shows a simple usage of the Logistic Regression class, this example classifies the positive and negative numbers (it is just an example)
import numpy as np
import logistic_regression_class as lrc

alpha=0.01 # learning rate
x=np.zeros((200,1))
y=np.zeros((200))

for i in range(100):
  x[i][0]=i
  y[i]=1

for i in range(101,200):
  x[i][0]=-i
  y[i]=0


b=0.0
n=y.shape[0] # number of training elements
m=x.shape[1] # number of properties
w=np.zeros((1,m))

test=lrc.logistic_regression(x,y,w,b,alpha)
test.train(120)
testcase=np.array([105])
## if the answer is above 0.5 so it is a positive number, else it is negative
print(lrc.sig(test.f(testcase)))

