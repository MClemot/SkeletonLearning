# -*- coding: utf-8 -*-

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def hessian(f, x):
    g = gradient(f(x), x)
    h0 = gradient(g[:,0], x)[:,None,:]
    h1 = gradient(g[:,1], x)[:,None,:]
    h2 = gradient(g[:,2], x)[:,None,:]
    h = torch.cat((h0,h1,h2), dim=1)
    return h