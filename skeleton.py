 # -*- coding: utf-8 -*-

import gudhi
import torch
from torch import nn
import numpy as np
from scipy.spatial import KDTree, Delaunay
from scipy.optimize import milp, LinearConstraint, Bounds

from tools import gradient

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sample_skeleton_gpu(net, pc, res, length, steps, div, orientation=-1):
    iso = pc.clone()
    number = iso.shape[0]
    c = torch.linspace(0., length, res, device=device)
    
    for k in range(steps):
        for d in range(div):  
            #sdf and gradient
            l_iso = iso[d*number//div:(d+1)*number//div,:]
            sdf = net(l_iso)
            gra = gradient(sdf, l_iso).detach()
            
            with torch.no_grad():
                #lines generation
                gra = nn.functional.normalize(orientation * gra, dim=1)
                pts = torch.mul(c[:,None,None], gra[None,:,:]) + l_iso
                sdf = net(pts)
            
                #in-shape lines generation
                val,ind = torch.max(((orientation*sdf[1:,:,:]) < 0.).float(), dim=0) #find first point outside
            
                bounds = c[ind] / length
                bounds += (bounds==0.)
                coefs = bounds @ c[None,:]
                pts = coefs[:,:,None] * gra[:,None,:] + l_iso[:,None,:]
            
            #finding smallest gradient if we search the lowest gradient's norm...
            pts.requires_grad = True
            sdf = net(pts)
            grn = gradient(sdf, pts).detach()
            
            with torch.no_grad():
                grn = torch.linalg.norm(grn, dim=2)
                arg = torch.argmin(grn, dim=1)
                # ...or the lowest SDF
                # arg = torch.argmin(sdf, dim=1)
            
                iso[d*number//div:(d+1)*number//div] = pts[torch.arange(l_iso.shape[0]), arg]
            
    sdf = net(iso).flatten()
    verif = torch.logical_and(orientation*sdf > 0.,
                              torch.linalg.norm(iso, ord=float('inf'), dim=1) <= 1.)
    iso = iso[verif,:]
    
    return iso.detach().cpu().numpy()


def reduce(m, reduce_radius):
    KD = KDTree(m)
    NN = KD.query_ball_tree(KD, reduce_radius)
    keep = [i for i in range(len(NN))]
    for i in range(len(NN)):
        if i in keep:
            for k in NN[i][1:]:
                if k>i and k in keep:
                    keep.remove(k)

    return m[keep]


def coverage_skeleton(skpts, pc, delta, factor=None, time_limit=120):
    n = skpts.shape[0]
    print(n, "candidates")
    
    L = torch.cdist(skpts, pc).detach().cpu().numpy()
    sdf = np.min(L, axis=1)[:,None]

    if factor==None:
        D = (L <= sdf + delta)
    else:
        D = (L <= (sdf * factor))


    D = np.transpose(D)

    #remove points that are not covered at the beginning (useful for siren and relu)
    npts = pc.shape[0]
    ind =  np.where(np.sum(D,axis=1) >= 1)
    pc = pc[ind]
    D = D[ind]
    nptsf = pc.shape[0]
    print(npts, " initial surface points; ",nptsf," filtered points")

    bounds = Bounds(lb=0, ub=1)
    constraints = LinearConstraint(D, lb=1)
    
    options = dict(disp=True, time_limit=time_limit, mip_rel_gap=.1)
    x = milp(np.ones((n)), integrality=1, bounds=bounds, constraints=constraints, options=options)
    
    cvskpts, cvskpts_sdf, triangles, edges = [], [], [], []
    
    if x.status >= 2:
        return np.array(cvskpts), edges, triangles

    for i in range(skpts.shape[0]):
        if x.x[i]>=.5:
            cvskpts.append(skpts[i].detach().cpu().numpy())
            cvskpts_sdf.append(sdf[i])
    cvskpts, cvskpts_sdf = np.array(cvskpts), np.array(cvskpts_sdf).flatten()
    n = cvskpts.shape[0]
    print(n, "coverage skeleton vertices")
    
    vertices = np.concatenate((cvskpts, pc.detach().cpu().numpy()))
    if factor == None:
        weights = np.concatenate((cvskpts_sdf + delta, np.zeros((pc.shape[0])) + delta))
    else:
        weights = np.concatenate((cvskpts_sdf*factor, np.zeros((pc.shape[0]))))
    
    alpha_complex = gudhi.alpha_complex.AlphaComplex(points=vertices, weights=list(weights**2))
    simplex_tree = alpha_complex.create_simplex_tree()
    
    for s in simplex_tree.get_simplices():
        if len(s[0]) == 3:
            t = s[0]
            if t[0] < n and t[1] < n and t[2] < n:
                triangles.append([t[0]+1, t[1]+1, t[2]+1])
        elif len(s[0]) == 2:
            e = s[0]
            if e[0] < n and e[1] < n:
                edges.append([e[0]+1, e[1]+1])
        
    return cvskpts, edges, triangles

def neural_candidates(skpts, reduce_radius):
    skpts = torch.tensor(reduce(skpts, reduce_radius), device=device).float()
    return skpts

def voronoi_candidates(pc, nc, reduce_radius):
    pc = pc.detach().cpu().numpy()
    nc = nc.detach().cpu().numpy()
    delaunay = Delaunay(pc)
    C = []
    for s in delaunay.simplices:
        v = [pc[s[0]], pc[s[1]], pc[s[2]], pc[s[3]]]
        A = np.stack((v[1]-v[0], v[2]-v[0], v[3]-v[0]))
        b = .5*np.array([np.linalg.norm(v[1])**2-np.linalg.norm(v[0])**2,
                         np.linalg.norm(v[2])**2-np.linalg.norm(v[0])**2,
                         np.linalg.norm(v[3])**2-np.linalg.norm(v[0])**2])
        cc = np.linalg.inv(A)@b
        is_inside = True
        for i in range(4):
            if np.dot(v[i]-cc, nc[s[i]])<0:
                is_inside = False
                break
        if is_inside:
            C.append(cc)
    
    C = np.array(C)
    C = reduce(C, reduce_radius)
    
    return torch.tensor(C, device=device).float()
