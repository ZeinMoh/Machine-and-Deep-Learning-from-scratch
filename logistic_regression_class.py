import numpy as np
import math

def sig(k):
  return 1/(1+math.exp(-k))

class logistic_regression:
  def __init__(self,xx,yy,ww,bb,a):
    self.x=xx
    self.y=yy
    self.alpha=a
    self.w=ww
    self.b=bb
    global n,m
    n=yy.shape[0]
    m=xx.shape[1]

  def f(self,xx):
    return (np.dot(self.w,xx.T)+self.b)[0]


  def J(self):
      total_err=0
      for i in range(n):
          est=sig(self.f(self.x[i]))
          total_err+= self.y[i]*math.log(sig(est))+(1-self.y[i])*math.log(1-sig(est))
      total_err/=(-n)
      return total_err

  def dJb(self):
      result=0.0
      for i in range(n):
          est= sig(self.f(self.x[i]))
          result+= (est-self.y[i])
      result*=2
      result/=n
      return result

  def dJw(self):
      final_result=np.zeros(m)
      for j in range(m):
          result=0.0
          for i in range(n):
              est= sig(self.f(self.x[i]))
              result+= (est-self.y[i])*self.x[i][j]
          result*=2
          result/=n
          final_result[j]=result
      return final_result

  def upd(self):
      self.b= self.b-self.alpha*self.dJb()
      djw=self.dJw()
      for i in range(m):
          self.w[0][i]= self.w[0][i]-self.alpha*djw[i]

  def train(self,t):
    while(t):
      self.upd()
      t-=1
