'''''
 a changed rnn using theano as background
'''
"""
Created on Wed Oct 25 21:52:22 2017

@author: freeze
"""

"""
do the vallina rnn using theano
"""
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt

#prepare data
data=open('D:\\input.txt','r').read()
chars=list(set(data))
data_size,vocab_size=len(data),len(chars)			#enconding as i of K
print 'data has %d character,%d unique.'%(data_size,vocab_size)
char_to_ix={ch:i for i,ch in enumerate(chars)}		#char_to_ix is like{'e:'0,'d':1,'h':2,...}
ix_to_char={i:ch for i,ch in enumerate(chars)}


sample_num=5
sample_dim=vocab_size
hidden_dim=50
output_dim=vocab_size
lr=0.02

x=T.matrix('x')
t=T.matrix('t')
h=T.vector('h')

wxh=theano.shared(np.random.random((sample_dim,hidden_dim))*0.1)
whh=theano.shared(np.random.random((hidden_dim,hidden_dim))*0.1)
why=theano.shared(np.random.random((hidden_dim,sample_dim))*0.1)
bh=theano.shared(0.)
by=theano.shared(0.)

def step(xi,hi,win,wh,wout,bhind,bout):
    h_t=T.tanh(T.dot(xi,win)+T.dot(hi,wh)+bhind)
    y_t=T.dot(h_t,wout)+bout
    y_p=T.exp(y_t)/T.exp(y_t).sum()     #probably
    return h_t,y_p

[hstate,yend],_up=theano.scan(step,sequences=x,
outputs_info=[h,None],non_sequences=[wxh,whh,why,bh,by])

lose=((yend-t)**2).sum()
#gradient
dwxh,dwhh,dwhy=T.grad(lose,[wxh,whh,why])
dbh,dby=T.grad(lose,[bh,by])

f=theano.function([x,t,h],[hstate,yend,lose],updates=[(wxh,wxh-lr*dwxh),(whh,whh-lr*dwhh),(why,why-lr*dwhy),(bh,bh-lr*dbh),(by,by-lr*dby)])



''''
 seting matrix data initination
'''
p=0
nt=0
seq_length=sample_num
itreation_steps=(data_size/vocab_size)*2000
h_zero=np.zeros(hidden_dim,) #initination state
 
while (nt<itreation_steps):
	if p+seq_length+1>=len(data):
		p=0
		h_zero=np.zeros(hidden_dim,)
	
	inputs_List=[]
	targets_List=[]
	inputs=[char_to_ix[ch] for ch in data[p:p+seq_length]]
	targets=[char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

	#print inputs,targets

	for i in inputs:
		inputs_seq=[0]*vocab_size
		inputs_seq[i]=1
		inputs_List.append(inputs_seq)
	
	for j in targets:
		targets_seq=[0]*vocab_size
		targets_seq[j]=1
		targets_List.append(targets_seq)

	input_matrix=np.array(inputs_List)
	target_matrix=np.array(targets_List)
	'''''
	do function
	'''
	answer=f(input_matrix,target_matrix,h_zero)
	print answer[2]
	h_zero=answer[0][-1]
	
	p+=seq_length
	nt+=1