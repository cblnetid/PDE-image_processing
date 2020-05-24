#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import autograd.numpy as np
from pylab import *
from autograd.util import *
from TVL_utils import *


def training_core(c,xi,yin,lambdas,tol,tau,eta):
    #implements the gradient descent time marching method for Total Variation learning
    #this is
    #lambda is the regularization parameter
    #c is the radial basis function parameter
    #tau is the step-size of the gradient descent method
    #tol is tolerance for the stopping criteria
    dim1,dim2 = xi.shape
    w=np.random.random((dim1,1))
    PSI=psi(xi,c,xi,w)
    w=np.linalg.inv(PSI.T.dot(PSI) + eta*np.identity(dim1)).dot(PSI.T.dot(yin))
    w=np.reshape(w,(dim1,1))

    nr=1
    i=0
    while nr > tol:
        if i==50:
            break
        i=i+1
        PSI=psi(xi,c,xi,w)
        DUDT=dudtv(c,xi,w,yin,lambdas)
        residual=np.linalg.inv(PSI.T.dot(PSI) + eta*np.identity(dim1)).dot(PSI.T.dot(DUDT))
        w = w + tau*residual
        nr=np.linalg.norm(residual)/len(w)
        #print('iter= %3.0i, rel.residual= %1.2e' % (i,nr))
    yout=psi(xi,c,xi,w).dot(w).T

    inds=np.where(yout > 0)
    yout[inds]=1
    inds=np.where(yout < 0)
    yout[inds]=-1
    return yout, w


# In[ ]:


def training_step(c,feature_training,lambdas,tol,tau,eta):
    #lambda is the regularization parameter
    #and c is the radial basis function parameter
    #tau is the step-size of the gradient descent method
    dim1,dim2 = feature_training.shape
    x=np.zeros((1,dim2-1))

    xi=np.zeros((dim1,dim2))
    xi=feature_training[0:dim1,1:dim2]
    #w=np.random.random((dim1,1))

    yin=np.zeros((dim1,1))
    yin[:,0]=feature_training[:,0]

    yout, w =training_core(c,xi,yin,lambdas,tol,tau,eta)

    Error=np.matrix.trace(yout!=yin)
    Efficiency=100-100*Error/dim1

    return yout, w, Error, Efficiency


# In[ ]:


def testing_step(feature_test,c,feature_training,w):
    #computes one testing step
    #c is the radial basis function parameter
    dim1,dim2 = feature_test.shape
    xt=np.zeros((dim1,dim2))
    xt=feature_test[0:dim1,1:dim2]

    yin=np.zeros((dim1,1))
    yin[:,0]=feature_test[:,0]

    dim1,dim2 = feature_training.shape
    xi=np.zeros((dim1,dim2))
    xi=feature_training[0:dim1,1:dim2]

    yout=psi(xt,c,xi,w).dot(w).T

    inds=np.where(yout > 0)
    yout[inds]=1
    inds=np.where(yout < 0)
    yout[inds]=-1

    dim1,dim2 = feature_test.shape
    Error=np.matrix.trace(yout!=yin)
    Efficiency=100-100*Error/dim1

    return yout, Error, Efficiency


# In[ ]:


def test_classifier(MAX,c,lambdas,eta,tol,mydata,upperbound,lowerbound,set_size,m,tau):
    #train and test the model
    #MAX is the number of epochs
    #eta is the levenberg marquardt parameter
    #lambda is the regularization parameter
    #and c is the radial basis function parameter
    #upperbound,lowerbound and set_size are used to split the dataset in training and testing sets
    test_eff= np.zeros((MAX))
    train_eff=np.zeros((MAX))
    for I in range(MAX):
        random_number = int(np.random.randint(upperbound+1,size=1))
        feature_test = mydata[random_number:random_number+set_size,:]
        feature_training = np.vstack(( mydata[0:random_number,:], mydata[random_number+set_size:m,:] ))

        yout, w, train_error, train_eff[I] = training_step(c,feature_training,lambdas,tol,tau,eta)
        yout, test_error, test_eff[I]= testing_step(feature_test,c,feature_training,w)

        inds=np.where(test_eff != 0)
        indst=inds[0]
        test_eff_mean=np.sum(test_eff[inds])/len(indst)
        print('Training_Eff = %2.2f, Testing_eff= %2.2f, Mean_Test_Eff= %2.2f' % (train_eff[I], test_eff[I], test_eff_mean))

        file1 = open("resultsTVfile.txt","a")
        s =  'c= ' + repr(c) + ', lambda= ' + repr(lambdas) + ', eta= ' + repr(eta) + ', trainef= ' + repr(train_eff[I]) + ', testef= ' + repr(test_eff[I])+ ', trainmeanef= ' + repr(test_eff_mean) + '\n'
        file1.write(s)
        file1.close()
    return test_eff_mean
