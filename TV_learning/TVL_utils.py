#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import autograd.numpy as np
from pylab import *
from autograd.util import *


def phi(x,c,xi):
    #computes phi(x) for each xi. In the paper is referred as $phi_i(x)$
    dim1,dim2 = xi.shape
    return np.reshape(np.exp(-0.5*c*np.linalg.norm(x-xi, axis=1)**2),(dim1,1))


# In[ ]:


def u(x,c,xi,w):
    #computes u(x) as in eqn (17)
    dim1,dim2 = xi.shape
    return np.sum(w*np.reshape(np.exp(-0.5*c*np.linalg.norm(x-xi, axis=1)**2),(dim1,1)))


# In[ ]:


def psi(x,c,xi,w):
    #computes capital psi as in page 3651 in the paper
    m=len(w)
    dim1,dim2 = x.shape
    M=np.zeros((dim1,m))
    for i in range(dim1):
        M[i,:]=np.reshape(phi(x[i],c,xi),m)
    return M


# In[ ]:


def g(x,c,xi,w):
    #computes g as in eqn (20) in the paper
    return np.sum(w*(x-xi)*phi(x,c,xi), axis=0)


# In[ ]:


def autograd_gradient_u(x,c,xi,w):
    #computes the gradient vector $\nabla \phi_i(x)$ using automatic differentiation
    nablau= np.grad(u,0)
    return nablau(x,c,xi,w)


# In[ ]:


def gradient_u(x,c,xi,w):
    #computes the gradient vector $\nabla \phi_i(x)$ as in the paper
    return -c*g(x,c,xi,w)


# In[ ]:


def laplacian_u(x,c,xi,w):
    #computes Laplacian of u as in the paper
    dim1,dim2 = xi.shape
    return c*np.sum(w*np.reshape((c*(np.linalg.norm(x-xi, axis=1)**2)-dim2),(dim1,1))*phi(x,c,xi))


# In[ ]:


def autograd_laplacian_u(x,c,xi,w):
    #computes Laplacian of u using automatic differentiation
    h=np.autograd_hessian_u(x,c,xi,w)
    return np.matrix.trace(h)


# In[ ]:


def autograd_hessian_u(x,c,xi,w):
    #computes the equivalent of the Hessian matriz in eqn (21) but using automatic differentiation
    dim1,dim2 = xi.shape
    def fun(x,c,xi,w):
        return np.sum(w*np.reshape(np.exp(-0.5*c*np.linalg.norm(x-xi, axis=1)**2),(dim1,1)))
    hess = np.hessian(fun)
    return np.reshape(hess(x,c,xi,w),(dim2,dim2))


# In[ ]:


def hessian_u(x,c,xi,w):
    #computes the Hessian matrix as in eqn (21) in the paper
    dim1,dim2 = xi.shape
    p=-c*np.sum(w*np.reshape(np.exp(-0.5*c*np.linalg.norm(x-xi, axis=1)**2),(dim1,1)))*np.identity(dim2)
    q=np.zeros((dim2,dim2))
    pp=phi(x,c,xi)
    for k in range(len(w)):
        q=q+w[k]*(x-xi[k]).T.dot((x-xi[k]))*pp[k]
    return p+(c**2)*q


# In[ ]:


def dudt(x,c,xi,w,y,lambdas):
    #computes each entry of the vector $\partial u / \partial t$ in the paper
    #you may decide to use the automatic differentiation functions for the gradient, Laplacian and Hessian operators
    uu=u(x,c,xi,w)
    gu=gradient_u(x,c,xi,w)
    hu=hessian_u(x,c,xi,w)
    lu=laplacian_u(x,c,xi,w)
    ngu=np.linalg.norm(gu)
    return np.asscalar(lambdas*(lu-(gu.dot(hu).dot(gu.T))/(gu.dot(gu.T)))/ngu - uu + y)


# In[ ]:


def dudtv(c,xi,w,y,lambdas):
    #computes the vector $\partial u / \partial t$
    m=len(w)
    dudtvv=np.zeros((m,1))
    for i in range(m):
        dudtvv[i,0]=dudt(xi[i],c,xi,w,y[i],lambdas)
    return dudtvv
