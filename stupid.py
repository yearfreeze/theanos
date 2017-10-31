# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 21:25:27 2017
@author: freeze
a course work
"""
import theano
import theano.tensor as T
import numpy as np
import random
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
"""
N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in xrange(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j
# lets visualize the data:
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.show()
"""
number=500
dimention=4
iteration=30000
learning_rate=0.9

x=T.matrix('x')
y=T.matrix('y')

w1=theano.shared(np.random.random((dimention,50)))
b1=theano.shared(0.)
w2=theano.shared(np.random.random((50,1)))
b2=theano.shared(0.)

h=T.tanh(T.dot(x,w1)+b1)
score=T.tanh(T.dot(h,w2)+b2)


loss=T.mean((y-score)**2)#loss function

dw1,dw2=T.grad(loss,[w1,w2])
db1,db2=T.grad(loss,[b1,b2])

f=theano.function([x,y],loss,updates=[(w1,w1-learning_rate*dw1),
(w2,w2-learning_rate*dw2),(b1,b1-learning_rate*db1),(b2,b2-learning_rate*db2)])
predict=theano.function([x],score)
#vector predict
"""
v=T.vector('v')
hidden=T.tanh(T.dot(v,w1[0])+b1)
ss=T.tanh(T.dot(hidden,w2)+b2)
pred=theano.function([x],ss)
"""


m_list=[]
for d in range(dimention):
    m_list.append(1.0/(d+1))
    
mu=np.array(m_list)
Sigma=np.identity(dimention)

R=cholesky(Sigma)
s1=np.dot(np.random.randn(number, dimention), R) + mu
t1=np.ones((number,1))
"""
plt.subplot(111)
#painting
plt.plot(s[:,0],s[:,1],'.')
plt.show()
"""
m_list=[]
for d in range(dimention):
    m_list.append(-1.0/(d+1))
    
mu=np.array(m_list)
Sigma=np.identity(dimention)

R=cholesky(Sigma)
s2=np.dot(np.random.randn(number, dimention), R) + mu
t2=np.zeros((number,1))-np.ones((number,1))
#transpored and add some random in design matrix
#would be better not used

sample_x=np.concatenate((s1,s2))
sample_t=np.concatenate((t1,t2))
"""
fuck=np.hstack((sample_x,sample_t))
ff=list(fuck)
random.shuffle(ff)
fy=np.array(ff)

sample_x=fy[:,0:dimention]
sample_t=fy[:,dimention:dimention+1]
"""
"""
begin to train by loop
"""

lost=[]
for i in range(iteration):
    lose=f(sample_x,sample_t)
    lost.append(lose)
    print lose
plt.plot(range(iteration),lost,label="loss function")

def account(r_score,r_target,number):
    count=0
    for s,t in zip(r_score,r_target):
        if abs(s-t)<0.1:
            count+=1
    return float(count)/number

def generate(source,arr):
    temp=np.zeros_like(source)
    for a in arr:
        temp[:,a]=source[:,a]
    return temp


test=predict(sample_x)
print 'accuracy whole: ',account(test,sample_t,number*2)
#one dimenstion testing
#test=sample_x[:,0]
#print do_as_vector(test)
test=predict(generate(sample_x,[0]))
print 'accuarcy only first dim: ',account(test,sample_t,number*2)

test=predict(generate(sample_x,[1]))
print 'accuarcy only second dim: ',account(test,sample_t,number*2)

test=predict(generate(sample_x,[2]))
print 'accuarcy only thrid dim: ',account(test,sample_t,number*2)

