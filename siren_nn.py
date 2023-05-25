# -*- coding: utf-8 -*-

import torch
import numpy as np
from torch import nn
from matplotlib import pyplot as plt

from tools import gradient

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# SIREN-style neural network
# =============================================================================

class Sine(nn.Module):
    def __init__(self, w0 = 30.):
        super().__init__()
        self.w0 = w0 
    def forward(self, x):
        return torch.sin(self.w0*x)   
    
class SirenLayer(nn.Module):
    def __init__(self, dim_in, dim_out, w0 = 1., c = 6., is_first = False, use_bias = True, activation = None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (np.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if bias is not None:
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        return self.activation(torch.nn.functional.linear(x, self.weight, self.bias))

class SirenNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, skip = [], w0 = 30., w0_initial = 30., activation = None): 
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.skip = [i in skip for i in range(num_layers)]

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            self.layers.append(SirenLayer(
                dim_in = layer_dim_in + (3 if self.skip[ind] else 0),
                dim_out = dim_hidden,
                w0 = layer_w0,
                use_bias = True,
                is_first = is_first,
                activation = activation
            ))
        self.last_layer = SirenLayer(dim_in = dim_hidden, dim_out = dim_out, w0 = w0, use_bias = True, activation = nn.Identity())

    def forward(self, x):
        i = x
        for k,layer in enumerate(self.layers):
            if not self.skip[k]:
                x = layer(x)
            else:
                x = layer(torch.concat((x,i), dim=-1))
        return self.last_layer(x)


# =============================================================================
# Losses functions
# =============================================================================
    
def sdf_loss_align(grad, normals):
    return (1-nn.functional.cosine_similarity(grad, normals, dim = 1)).mean()

# =============================================================================
# Optimization
# =============================================================================

def optimize_siren_neural_sdf(net, optim, pc, nc, batch_size, pc_batch_size,
                        epochs, lambda_pc = 100, lambda_eik=200,
                        lambda_f=100, plot_loss=False):
    
    lpc, leik, lf = [], [], []
    
    def evaluate_loss():
        nonlocal firsteval
        
        pts_random = torch.rand((batch_size, 3), device = device)*2-1
        pts_random.requires_grad = True
      
        #predict the sdf and gradients for all points
        if pc_batch_size != None:
            sample = torch.randint(pc.shape[0], (pc_batch_size,))
        sample_pc = pc if pc_batch_size == None else pc[sample]
        sample_nc = nc if pc_batch_size == None else nc[sample]
        sdf_pc = net(sample_pc)
        sdf_random = net(pts_random)
        grad_pc = gradient(sdf_pc, sample_pc)
        grad_random = gradient(sdf_random, pts_random)
      
        # compute and store the losses
      
        loss_pc = 100*nn.functional.mse_loss(sdf_pc, torch.zeros_like(sdf_pc)) + sdf_loss_align(grad_pc, sample_nc)
        loss_f =  torch.exp(-1e2*torch.abs(sdf_random)).mean()     
        loss_eik = nn.functional.mse_loss(grad_random.norm(dim=1), torch.ones((batch_size), device=device))
        
        # append all the losses
        if firsteval:
            lf.append(float(loss_f))
            lpc.append(float(loss_pc))
            leik.append(float(loss_eik))
            firsteval = False
      
        # sum the losses of reach of this set of points
        loss = lambda_pc*loss_pc + lambda_eik*loss_eik + lambda_f*loss_f
        optim.zero_grad()
        loss.backward()
        
        return loss
    
    for batch in range(epochs):
        firsteval = True
        optim.step(evaluate_loss)

        ## display the result
        #if plot_loss & (batch%50 == 49 if isinstance(optim,torch.optim.LBFGS) else batch%5000==4999):
        #    plt.figure(figsize=(6,4))
        #    plt.yscale('log')
        #    plt.plot(lpc, label = 'Point cloud loss ({:.2f})'.format(lpc[-1]))
        #    plt.plot(leik, label = 'Eikonal loss ({:.2f})'.format(leik[-1]))
        #    plt.xlabel("Epochs")
        #    plt.legend()
        #    plt.show()

    # display the result
    if plot_loss:
        plt.figure(figsize=(6,4))
        plt.yscale('log')
        plt.plot(lpc, label = 'Point cloud loss ({:.2f})'.format(lpc[-1]))
        plt.plot(leik, label = 'Eikonal loss ({:.2f})'.format(leik[-1]))
        plt.plot(lf, label = 'Faraway loss ({:.2f})'.format(lf[-1]))
        plt.xlabel("Epochs")
        plt.legend()
        plt.show()
        #plt.savefig("skel7clean_loss.pdf")
        #plt.close()

def pretrain(dim_hidden, num_layers, skip, lr, batch_size, epochs, activation='Sine'):
    if activation == 'Sine':
        net = SirenNet(
            dim_in = 3,
            dim_hidden = dim_hidden,
            dim_out = 1,
            num_layers = num_layers,
            skip = skip,
            w0_initial = 30.,
            w0 = 30.,
            ).to(device)
    elif activation == 'ReLU':
        net = SirenNet(
            dim_in = 3,
            dim_hidden = dim_hidden,
            dim_out = 1,
            num_layers = num_layers,
            skip = skip,
            activation = nn.functional.relu,
            w0_initial = 1.,
            w0 = 1.,
            ).to(device)
    elif activation == 'SoftPlus':
        net = SirenNet(
            dim_in = 3,
            dim_hidden = dim_hidden,
            dim_out = 1,
            num_layers = num_layers,
            skip = skip,
            activation = nn.functional.softplus,
            w0_initial = 1.,
            w0 = 1.,
            ).to(device)
    
    optim = torch.optim.Adam(lr=lr, params=net.parameters())
    
    lpc, loth = [], []
    
    try:
        for batch in range(epochs):
            pts_random = torch.rand((batch_size, 3), device = device)*2-1
            pts_random.requires_grad = True
            
            pred_sdf_random = net(pts_random)
            
            gt_sdf_random = torch.linalg.norm(pts_random, dim=1) - 0.5
            loss_pc = nn.functional.mse_loss(pred_sdf_random.flatten(), gt_sdf_random) * 1e1
            
            grad_random = gradient(pred_sdf_random, pts_random)    
            loss_other = nn.functional.mse_loss(grad_random.norm(dim=1), torch.ones((batch_size), device=device))
            
            # append all the losses
            lpc.append(float(loss_pc))
            loth.append(float(loss_other))
          
            # sum the losses of reach of this set of points
            loss = loss_pc + loss_other
            optim.zero_grad()
            loss.backward()
          
            optim.step()
          
            # display the result
            if batch%50 == 49:
                plt.figure(figsize=(8,6))
                plt.yscale('log')
                plt.plot(lpc, label = f'Point cloud loss ({lpc[-1]})')
                plt.plot(loth, label = f'Other points loss ({loth[-1]})')
                plt.legend()
                plt.show()
    except KeyboardInterrupt:
        pass
    
    return net
