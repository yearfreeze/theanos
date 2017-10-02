# -*- coding: utf-8 -*-
"""
Created on Sun Oct 01 20:03:01 2017

@author: freeze
"""

import theano
import theano.tensor as T
import numpy as np
import time
import matplotlib.pyplot as plt

x_train=np.array([[0,0],[0,1],[1,0],[1,1]])
y_train=np.array([[0],[1],[1],[0]]) #y error
#print x_train,y_train

#construct graph
x=T.matrix('x')
y=T.matrix('y')

w1=theano.shared(np.random.rand(2,2))
b1=theano.shared(0.)
w2=theano.shared(np.random.rand(2,1))
b2=theano.shared(0.)

#w1_print=theano.printing.Print('the w1 is',w1)

h1=1/(1+T.exp(-T.dot(x,w1)-b1))
h2=1/(1+T.exp(-T.dot(h1,w2)-b2))

loss=T.mean((y-h2)**2)  #loss function


gw1,gw2=T.grad(loss,[w1,w2])
gb1,gb2=T.grad(loss,[b1,b2])

start=time.time()

f=theano.function([x,y],[loss],updates=[(w1,w1-0.5*gw1),(w2,w2-0.5*gw2),(b1,b1-0.5*gb1),(b2,b2-0.5*gb2)])
predict=theano.function([x],h2)
#show=theano.function([],w1_print)

itreation_steps=10000 # step
lost=[]
for i in range(itreation_steps):
    lose=f(x_train,y_train)
    lost.append(lose)
    print lose

plt.plot(range(itreation_steps),lost,label="first")
end=time.time()
#show()
print 'total using time',end-start