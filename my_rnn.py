''''
character-level Rnn model
'''
import numpy as np
import time
import matplotlib.pyplot as plt

#data IO
data=open('D:\\input.txt','r').read()
chars=list(set(data))
data_size,vocab_size=len(data),len(chars)
print 'data has %d character,%d unique.'%(data_size,vocab_size)
char_to_ix={ch:i for i,ch in enumerate(chars)}
ix_to_char={i:ch for i,ch in enumerate(chars)}

#hyperparmeters
itreation_steps=5000
hidden_size=10
seq_length=5
learning_rate=1e-1

#model paramters
Wxh=np.random.randn(hidden_size,vocab_size)*0.01
Whh=np.random.randn(hidden_size,hidden_size)*0.01
Why=np.random.randn(vocab_size,hidden_size)*0.01
bh=np.zeros((hidden_size,1))
by=np.zeros((vocab_size,1))

def lossFun(inputs,targets,hprev):
	''''
	inputs,targets are both list of integers.
	hprev is H*1 array of initial hidden state.
	returns the loss,gradients on model paramters ,and last hidden state
	'''
	xs,hs,ys,ps={},{},{},{}
	hs[-1]=np.copy(hprev)
	loss=0
	#forward pass
	for t in xrange(len(inputs)):
		xs[t]=np.zeros((vocab_size,1))  #encode in 1-of-k 
		xs[t][inputs[t]]=1
		hs[t]=np.tanh(np.dot(Wxh,xs[t])+np.dot(Whh,hs[t-1])+bh)
		ys[t]=np.dot(Why,hs[t])+by
		ps[t]=np.exp(ys[t])/np.sum(np.exp(ys[t])) #probabilities for next char
		loss+= -np.log(ps[t][targets[t],0])       #softmax                      find the target/s value in dict(ps[t]) 
		
	#backword pass:compute gradients
	dWxh,dWhh,dWhy=np.zeros_like(Wxh),np.zeros_like(Whh),np.zeros_like(Why)
	dbh,dby=np.zeros_like(bh),np.zeros_like(by)
	dhnext=np.zeros_like(hs[0])
	
	for t in reversed(xrange(len(inputs))):		#reversed output the xrange number
		dy=np.copy(ps[t])
		dy[targets[t]]-=1
		dWhy+=np.dot(dy,hs[t].T)
		dby+=dy
		dh=np.dot(Why.T,dy)+dhnext
		dhraw=(1-hs[t]*hs[t])*dh			#derive of tanh is 1-tanh**2
		dbh+=dhraw
		dWxh+=np.dot(dhraw,xs[t].T)
		dWhh+=np.dot(dhraw,hs[t-1].T)
		dhnext=np.dot(Whh.T,dhraw)
	for dparam in [dWxh,dWhh,dWhy,dbh,dby]:
		np.clip(dparam,-5,5,out=dparam)		#clip to mitigate exploding gradients
	return loss,dWxh,dWhh,dWhy,dbh,dby,hs[len(inputs)-1]

def sample(h,seed_ix,n):
	''''
	sample a sequence of intergers from the model
	h is meomry state ,seed_ix is seed letter for first time step
	'''
	x=np.zeros((vocab_size,1))
	x[seed_ix]=1
	ixes=[]
	for t in xrange(n):
		h=np.tanh(np.dot(Wxh,x)+np.dot(Whh,h)+bh)
		y=np.dot(Why,h)+by
		p=np.exp(y)/np.sum(np.exp(y))
		ix=np.random.choice(range(vocab_size),p=p.ravel())	#random choice a number ,ravel change lie to hang
		x=np.zeros((vocab_size,1))
		x[ix]=1
		ixes.append(ix)
	return ixes

n,p=0,0
hprev=np.zeros((hidden_size,1))

start=time.time()
lost=[]

while n<itreation_steps:
	#prepare inputs 
	if p+seq_length+1>=len(data) or n==0:   # /vocab_size:
		hprev=np.zeros((hidden_size,1))
		p=0
	inputs=[char_to_ix[ch] for ch in data[p:p+seq_length]]
	targets=[char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]
	
	#forward seq_length characters through the net and fetch gradients
	loss,dWxh,dWhh,dWhy,dbh,dby,hprev=lossFun(inputs,targets,hprev)
	
	if n%2==0:
		sample_ix=sample(hprev,inputs[0],10)
		txt=' '.join(ix_to_char[ix] for ix in sample_ix)
		#txt=' '+(ix_to_char[ix] for ix in sample_ix)
		print 'iter %d,loss:%f'%(n,loss)
		print '____\n %s \n____'%(txt,)
	lost.append(loss)  #to paint
	#update paramters
	Wxh-=learning_rate*dWxh
	Whh-=learning_rate*dWhh
	Why-=learning_rate*dWhy
	bh-=learning_rate*dbh
	by-=learning_rate*dby
	
	p+=seq_length
	n+=1
	
plt.plot(range(n),lost,label="rnn")
end=time.time()
#show()
print 'total using time',end-start