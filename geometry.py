# -*- coding: utf-8 -*-

import torch
import numpy as np
from pygel3d import hmesh
from scipy.spatial import KDTree

from tools import gradient

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# bounding boxes
# =============================================================================

def get_bounding_box(pc):
    x1,x2 = torch.min(pc[:,0]), torch.max(pc[:,0])
    y1,y2 = torch.min(pc[:,1]), torch.max(pc[:,1])
    z1,z2 = torch.min(pc[:,2]), torch.max(pc[:,2])
    m = torch.tensor([.5*(x1+x2), .5*(y1+y2), .5*(z1+z2)], device=device)
    scale = torch.max(torch.abs(pc-m))+0.01
    return m, scale


def center_bounding_box(pc):
    x1,x2 = torch.min(pc[:,0]), torch.max(pc[:,0])
    y1,y2 = torch.min(pc[:,1]), torch.max(pc[:,1])
    z1,z2 = torch.min(pc[:,2]), torch.max(pc[:,2])
    m = torch.tensor([.5*(x1+x2), .5*(y1+y2), .5*(z1+z2)], device=device)
    pc -= m
    scale = torch.max(torch.abs(pc))+0.01
    pc /= scale
    return pc, m, scale


# =============================================================================
# generation
# =============================================================================

def cube_point_cloud(c, pts_per_side, device):
    pts = torch.rand((pts_per_side, 6, 3))
    pts = pts * torch.tensor([[[c, c, 0], [c, 0, c], [0, c, c], [c, c, 0], [c, 0, c], [0, c, c]]]) + .5 * c * torch.tensor([[[-1, -1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]]) 
    normals = torch.ones_like(pts) * torch.tensor([[[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, -1], [0, -1, 0], [-1, 0, 0]]])
    return pts.reshape(-1, 3).to(device), normals.reshape(-1, 3).to(device)


# =============================================================================
# geometry processing
# =============================================================================

def projection(net, iterations=10, isopoints=None, number=1000, prune=False):
    #initializing iso-points if needed
    if isopoints == None:
        isopoints = torch.rand((number, 3), device = device)*1.9-.95
    if not isopoints.requires_grad:
        isopoints.requires_grad = True
    
    for it in range(iterations):
        isopoints_sdf = net(isopoints)
        grad = gradient(isopoints_sdf, isopoints)
        inv = grad / (torch.norm(grad, dim=1)**2)[:,None]
        isopoints = isopoints - inv * isopoints_sdf
        
    if prune:
        keep = torch.linalg.norm(isopoints, ord=float('inf'), dim=1) <= 1.
        isopoints = isopoints[keep,:]
  
    return isopoints

def uniform_resampling(net, isopoints, steps, K, alpha, sigma, boundary=False):
    
    for step in range(steps):       
        isopoints_sdf = net(isopoints)
        grad = gradient(isopoints_sdf, isopoints).detach()
        
        with torch.no_grad():
            iso_cpu = isopoints.detach().cpu().numpy()
            KDT = KDTree(iso_cpu)
            NNv, NNi = KDT.query(iso_cpu, k=K+1)
            NNv = torch.tensor(NNv, device=device).float()[:,1:]
        
            # we compute with the KNN graph the wanted shift for each isopoint
            knn = isopoints[NNi[:,1:]]
            direc = knn - isopoints[:,None,:]
            
            # projection on the tangent plane
            grad = torch.nn.functional.normalize(grad)
            grad = grad.repeat(K,1,1).transpose(0,1)
            direc = direc - torch.linalg.vecdot(direc, grad)[:,:,None] * grad
            
            #weighting, summing
            direc = torch.nn.functional.normalize(direc, dim=2)
            direc *= torch.exp(-NNv**2/sigma)[:,:,None]
            r = torch.sum(direc, dim=1)
            
            #we move the isopoints
            isopoints = isopoints - alpha*r
            if boundary:
                isopoints = torch.maximum(-torch.ones((1), device=device), torch.minimum(torch.ones((1), device=device), isopoints))

        isopoints = projection(net, iterations=2, isopoints=isopoints, prune=True)

        #KDT = KDTree(iso_cpu)
        #NNv, NNi = KDT.query(iso_cpu, k=K+1)
        #NNv = torch.tensor(NNv, device=device).float()[:,1:]

        #ind = NNv[:,1] < 0.02
        #isopoints = isopoints[ind,:]
        
    return isopoints


# =============================================================================
# sampling on mesh
# =============================================================================

def sample_mesh(s, N):
    m = hmesh.load(s)
    sample = []
    normals = []
    
    M = 0
    for v in m.positions():
        M = max(M, np.linalg.norm(v, np.inf))
    
    area = 0
    for t in m.faces():
        area += m.area(t)
    
    for t in m.faces():
        n = m.face_normal(t)
        ver = []
        for v in m.circulate_face(t, mode='v'):
            ver.append(v)
        r = m.area(t)/area * N
        if r<1:
            if np.random.rand() > r:
                continue
        num = int(np.ceil(r))
        r1 = np.random.rand(num)
        r2 = np.random.rand(num)
        loc = (np.ones_like(r1)-np.sqrt(r1))[:,None] * m.positions()[ver[0]][None,:]
        loc += (np.sqrt(r1)*(np.ones_like(r1)-r2))[:,None] * m.positions()[ver[1]][None,:]
        loc += (np.sqrt(r1)*r2)[:,None] * m.positions()[ver[2]][None,:]
        for p in loc:
            sample.append(p)
            normals.append(n)
    
    pc = 1/M * torch.tensor(np.array(sample), device=device).float()
    pc.requires_grad = True
    nc = torch.tensor(np.array(normals), device=device).float()
    return pc, nc, M
