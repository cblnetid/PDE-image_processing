#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import autograd.numpy as np
from pylab import *
from autograd.util import *
from learning_functions import *
from TVL_utils import *

#load dataset
mydata2 = np.loadtxt('bupaNormalized.dat')
m,n = mydata2.shape
mydata = np.hstack ((mydata2[:,6].reshape(m,1), mydata2[:,0:6]))

#N-fold crossvalidation
N=5
set_size=int(round(m/N))
upperbound=m-set_size
lowerbound=set_size

#number of training epochs
MAX=10

#parameters search vector
cc=np.array([0.1, 0.2, 0.5, 1.0, 2.0])
lambdass=np.array([0.001, 0.01, 0.1, 1.0, 10.0, 100])
etas=np.array([.001])

#stopping criteria
tol=0.005
#gradient descent step-size
tau=0.1

#define name of file for the saving results
file1 = open("resultsTVfile.txt","w")
file1.write("Results for TV model - May 23, 2020 \n")
file1.close()

#train
for c in cc:
    for lambdas in lambdass:
        for eta in etas:
            file1 = open("resultsTVfile.txt","a")
            s =  'processing with c= ' + repr(c) + ', lambda= ' + repr(lambdas) + ', eta= ' + repr(eta) + '\n'
            file1.write(s)
            file1.close()
            print('Processing with lambda= %1.3f, c= %2.3f, eta= %1.3f, tol= %1.3f' % (lambdas,c,eta,tol))
            test_classifier(MAX,c,lambdas,eta,tol,mydata,upperbound,lowerbound,set_size,m,tau)
