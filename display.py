# -*- coding: utf-8 -*-

import torch
import numpy as np
from matplotlib import pyplot as plt

from tools import gradient

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def display_sdfColor(f, resolution, z, axis='z', filename = None, matrix=None):
    """
    displays the values of the function f, evaluated over a regular grid defined between -1 and 1 and of resolution (resolution x resolution)
    """
    coords = torch.stack(list(torch.meshgrid([torch.linspace(-1, 1, resolution, device = device)]*2, indexing = 'xy')), dim=2).reshape(-1, 2)
    if axis == 'x':
        coords = torch.concat((torch.zeros((coords.shape[0],1), device=device)+z, coords), dim=1)
    elif axis == 'y':
        coords = torch.concat((coords[:,:1], torch.zeros((coords.shape[0],1), device=device)+z, coords[:,1:]), dim=1)
    elif axis == 'z':
        coords = torch.concat((coords, torch.zeros((coords.shape[0],1), device=device)+z), dim=1)

    if matrix != None :
        matrix = torch.transpose(matrix,0,1)
        coords = torch.matmul(coords,matrix.cuda())

    coords.requires_grad = True
    sdf = f(coords).reshape(resolution, resolution)
    numpy_sdf = sdf.detach().cpu().numpy()

    eps = 0.005
    numpy_sdf_max = np.ones(numpy_sdf.shape)-np.maximum(numpy_sdf,np.zeros(numpy_sdf.shape))
    numpy_sdf_max = numpy_sdf_max - np.multiply(numpy_sdf_max, np.multiply(numpy_sdf<=eps, numpy_sdf>=-eps))
    numpy_sdf_min = np.ones(numpy_sdf.shape)-np.maximum(-numpy_sdf,np.zeros(numpy_sdf.shape))
    numpy_sdf_min = numpy_sdf_min - np.multiply(numpy_sdf_min, np.multiply(numpy_sdf<=eps, numpy_sdf>=-eps))
    numpy_sdf_both = np.ones(numpy_sdf.shape)-np.maximum(numpy_sdf,np.zeros(numpy_sdf.shape))-np.maximum(-numpy_sdf,np.zeros(numpy_sdf.shape))
    numpy_sdf_both = numpy_sdf_both - np.multiply(numpy_sdf_both, np.multiply(numpy_sdf<=eps, numpy_sdf>=-eps))

    plt.axis('off')
    plt.imshow(np.concatenate([numpy_sdf_min[:,:,np.newaxis],numpy_sdf_both[:,:,np.newaxis],numpy_sdf_max[:,:,np.newaxis]], axis = 2) )
    plt.contour(numpy_sdf, 10, colors='k', linewidths=.4, linestyles='solid')
    if filename==None:
        plt.show()
    else :
        plt.savefig(filename)
        plt.close()

def display_grad(f, resolution, z, axis='z',filename=None, matrix=None):
    """
    displays the norm of the gradient of the function f, evaluated at points of a regular grid defined between -1 and 1 and of resolution (resolution x resolution)
    """
    coords = torch.stack(list(torch.meshgrid([torch.linspace(-1, 1, resolution, device = device)]*2, indexing = 'xy')), dim=2).reshape(-1, 2)
    if axis == 'x':
        coords = torch.concat((torch.zeros((coords.shape[0],1), device=device)+z, coords), dim=1)
    elif axis == 'y':
        coords = torch.concat((coords[:,:1], torch.zeros((coords.shape[0],1), device=device)+z, coords[:,1:]), dim=1)
    elif axis == 'z':
        coords = torch.concat((coords, torch.zeros((coords.shape[0],1), device=device)+z), dim=1)

    if matrix != None :
        matrix = torch.transpose(matrix,0,1)
        coords = torch.matmul(coords,matrix.cuda())

    coords.requires_grad = True
    sdf = f(coords)
    grad = gradient(sdf, coords).norm(dim = 1).detach().cpu().numpy().reshape(resolution, resolution)
    
    plt.axis('off')
    plt.imshow(grad, cmap = "nipy_spectral", vmin = 0., vmax = 1.5)  # 0.25, 1.25
    plt.colorbar()
    if filename==None:
        plt.show()
    else :
        plt.savefig(filename)
        plt.close()


def display_gradgrad(f, resolution, z, axis='z', filename = None, matrix=None):
    """
    displays the norm of the gradient of the norm of the gradient of the function f, evaluated at points of a regular grid defined between -1 and 1 and of resolution (resolution x resolution)
    """
    coords = torch.stack(list(torch.meshgrid([torch.linspace(-1, 1, resolution, device = device)]*2, indexing = 'xy')), dim=2).reshape(-1, 2)
    if axis == 'x':
        coords = torch.concat((torch.zeros((coords.shape[0],1), device=device)+z, coords), dim=1)
    elif axis == 'y':
        coords = torch.concat((coords[:,:1], torch.zeros((coords.shape[0],1), device=device)+z, coords[:,1:]), dim=1)
    elif axis == 'z':
        coords = torch.concat((coords, torch.zeros((coords.shape[0],1), device=device)+z), dim=1)

    if matrix != None :
        matrix = torch.transpose(matrix,0,1)
        coords = torch.matmul(coords,matrix.cuda())

    coords.requires_grad = True
    sdf = f(coords)
    grad = gradient(sdf, coords).norm(dim=1)
    grad2 = gradient(grad, coords).norm(dim=1).detach().cpu().numpy().reshape(resolution, resolution)
    
    plt.axis('off')
    plt.imshow(grad2, cmap = "plasma")#, vmin = 0, vmax = 20)
    plt.colorbar()
    if filename==None:
        plt.show()
    else :
        plt.savefig(filename)
        plt.close()
