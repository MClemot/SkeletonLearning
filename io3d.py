# -*- coding: utf-8 -*-

import torch
import numpy as np
from pygel3d import hmesh

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def to_obj(pts, path, factor=1., tri = None, lines = None):
    f = open(path, "w")
    for p in pts:
        f.write("v {} {} {}\n".format(factor*p[0], factor*p[1], factor*p[2]))
    if tri != None:
        for t in tri:
            l = list(t)
            f.write("f {} {} {}\n".format(l[0], l[1], l[2]))
    if lines != None:
        for li in lines:
            l = list(li)
            f.write("l {} {}\n".format(l[0], l[1]))
    f.close()

def from_obj(path):
    f = open(path, "r")
    s = f.readline()
    L = []
    while s:
        t = s.split()
        L.append(np.array([float(t[1]), float(t[2]), float(t[3])]))
        s = f.readline()
    f.close()
    return np.array(L)

def xyz_to_obj(xyzfile, objfile):
    f1 = open(xyzfile, "r")
    s = f1.readline()
    L = []
    while s:
        t = s.split()
        L.append(np.array([float(t[0]), float(t[1]), float(t[2])]))
        s = f1.readline()
    f1.close()

    f = open(objfile, "w")
    for p in L:
        f.write("v {} {} {}\n".format(p[0], p[1], p[2]))
    f.close()
   

def from_ply(path):
    f = open(path, "r")
    s = f.readline()
    while s[:-1] != "end_header":
        s = f.readline()
    s = f.readline()
    p, n = [], []
    while s:
        t = s.split()
        p.append(np.array([float(t[0]), float(t[1]), float(t[2])]))
        n.append(np.array([float(t[3]), float(t[4]), float(t[5])]))
        s = f.readline()
    f.close()
    return np.array(p), np.array(n)

def from_xyz(path):
    f = open(path, "r")
    s = f.readline()
    L = []
    while s:
        t = s.split()
        L.append(np.array([float(t[0]), float(t[1]), float(t[2])]))
        s = f.readline()
    f.close()
    return np.array(L)

def from_xyz_normals(path):
    f = open(path, "r")
    s = f.readline()
    L = []
    n = []
    while s:
        t = s.split()
        L.append(np.array([float(t[0]), float(t[1]), float(t[2])]))
        n.append(np.array([float(t[3]), float(t[4]), float(t[5])]))
        s = f.readline()
    f.close()
    return np.array(L), np.array(n)

def from_skel(path):
    f = open(path, "r")
    s = f.readline()
    while s:
        t = s.split()
        if len(t):
            if t[0] == "CN":
                k = int(t[1])
                break
        s = f.readline()
        
    L = []
    E = []
    
    for i in range(k):
        t = f.readline().split()
        n = int(t[1])
        for j in range(n):
            t = f.readline().split()
            L.append(np.array([float(t[0]), float(t[1]), float(t[2])]))
            if j:
                E.append([len(L)-1, len(L)])
    
    f.close()
    return np.array(L), E

def to_xyz_normals(path, pc, nc):
    f = open(path, "w")
    for i in range(0,pc.shape[0]):
        f.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(pc[i,0], pc[i,1], pc[i,2], nc[i,0], nc[i,1], nc[i,2]))
    f.close()

def load_mesh(s, normalize=True):
    m = hmesh.load(s)

    normals = []
    for v in m.vertices():
        normals.append(m.vertex_normal(v))
    normals = np.array(normals)
    
    M = 0
    for v in m.positions():
        M = max(M, np.linalg.norm(v, np.inf))
    
    pts_point_cloud, normal_point_cloud = (1/M if normalize else 1)*torch.tensor(m.positions(), device=device).float(), torch.tensor(normals, device=device).float()
    pts_point_cloud.requires_grad = True
    
    return pts_point_cloud, normal_point_cloud, M


