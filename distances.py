# -*- coding: utf-8 -*-

import numpy as np
from pygel3d import hmesh
from scipy.spatial import KDTree
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import KDTree

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compare_skeletal_points_to_gtskel(groundtruth, skelpts, mindist = 0.01, verbose=False):
    vertices, edges, faces = readobj(groundtruth)
    samples1, edges1, faces1 = readobj(skelpts)

    gtsamples = sample_skeleton(vertices, edges, faces, mindist)
    
    cd = chamfer(samples1, gtsamples)
    hd = hausdorff(samples1, gtsamples)

    sdcd = 0
    sdhd = 0
    return cd,hd


def compare_skeletal_points_to_gtpts(samplesgt, skelpts, mindist = 0.01, verbose=False):
    samples1, edges1, faces1 = readobj(skelpts)

    cd = chamfer(samples1, samplesgt)
    hd = hausdorff(samples1, samplesgt)

    sdcd = 0
    sdhd = 0
    return cd,hd
    


def compare_skeletons(objfile1, objfile2, mindist = 0.05, verbose=False):
    vertices1, edges1, faces1 = readobj(objfile1)
    vertices2, edges2, faces2 = readobj(objfile2)

    samples1 = sample_skeleton(vertices1, edges1, faces1, mindist)
    samples2 = sample_skeleton(vertices2, edges2, faces2, mindist)

    if verbose:
        print("file ",objfile1," sampled with ", np.size(samples1,0)," samples.")
        print("file ",objfile2," sampled with ", np.size(samples2,0)," samples.")

    
    cd = chamfer(samples1, samples2)
    hd = hausdorff(samples1, samples2)

    sdcd = 0
    sdhd = 0
    #sdh1 = semi_discrete_directed_hausdorff(samples1, vertices2, edges2, faces2)
    #sdh2 = semi_discrete_directed_hausdorff(samples2, vertices1, edges1, faces1)

    #sdhd = max(sdh1, sdh2)

    #sdc1 = semi_discrete_chamfer(samples1, vertices2, edges2, faces2)
    #sdc2 = semi_discrete_chamfer(samples2, vertices1, edges1, faces1)

    #sdcd = 0.5*(sdc1+sdc2)

    #if verbose:
    #    print("Comparing samplings of the skeletons")
    #    print("Chamfer distance: ", cd)
    #    print("Hausdorff distance: ", hd)
    
    #    print("Comparing skeletons using semi-discrete distances")
    #    print("SD Chamfer distance: ", sdcd)
    #    print("SD Hausdorff distance: ", sdhd)

    #return cd, hd, sdcd, sdhd
    return sdcd, sdhd, cd, hd


def sample_skeleton(vertices, edges, faces, mindist):
    
    sqmindist = mindist**2

    samples = []
    i = 0
    for t in faces:
        v0 = vertices[t[0] - 1,:]
        v1 = vertices[t[1] - 1,:]
        v2 = vertices[t[2] - 1,:]
        area = 0.5 * np.linalg.norm( np.cross( v1 - v0 , v2 - v0 ))
        r = area/sqmindist;
        
        num = int(np.ceil(r))
        r1 = np.random.rand(num)
        r2 = np.random.rand(num)

        loc = (np.ones_like(r1) - np.sqrt(r1))[:,None] * v0 + (np.sqrt(r1)*(np.ones_like(r1)-r2))[:,None] * v1 + (np.sqrt(r1)*r2)[:,None] * v2
        for p in loc:
            samples.append(p)
        i = i+1

    for e in edges:
        v0 = vertices[e[0] - 1,:]
        v1 = vertices[e[1] - 1,:]
        area = np.linalg.norm(v1 - v0)
        r = area/mindist;

        num = int(np.ceil(r))
        r1 = np.random.rand(num)
        loc = r1[:,None] * v1 + (np.ones_like(r1) - r1)[:,None] * v0
        for p in loc:
            samples.append(p)

    for v in vertices:
        samples.append(v)

    print(len(samples))    
    return samples

def readobj(objfile):
    obj = open(objfile, 'r')


    verticeslist = []
    edgeslist = []
    faceslist = []

    for line in obj:
        split = line.split()
        if not len(split):
            continue
        if split[0] == "v":
            verticeslist.append([np.double(x) for x in split[1:4]])
        elif split[0] == "f":
            faceslist.append([int(x) for x in split[1:]])
        elif split[0] == "l":
            edgeslist.append([int(x) for x in split[1:]])

    if not len(verticeslist):
        return None
    
    vertices = np.asarray(verticeslist)
    faces = np.asarray(faceslist)
    edges = np.asarray(edgeslist)

    return vertices, edges, faces

def chamfer(X,Y):
    KDTx = KDTree(X)
    distx, idx = KDTx.query(Y, k=1)
    KDTy = KDTree(Y)
    disty, idx = KDTy.query(X, k=1)

    distx = np.mean(np.square(distx))
    disty = np.mean(np.square(disty))
    
    return 0.5*(distx + disty)


def hausdorff(X,Y):
    #KDTx = KDTree(X)
    #distx, idx = KDTx.query(Y, k=1)
    #h1 = np.amax(distx,0)
    #KDTy = KDTree(Y)
    #disty, idx = KDTy.query(X, k=1)
    #h2 = np.amax(disty,0)
    h1 = directed_hausdorff(X,Y)[0]
    h2 = directed_hausdorff(Y,X)[0]
    return np.maximum(h1,h2)


def distance_to_vertices(samples, vertices):
    KDT = KDTree(vertices)
    NNv = KDT.query(samples, 1)[0]
    return NNv

def distance_to_edge(samples, v0, v1):
    length = np.linalg.norm(v1 - v0)
    dir = (v1 - v0) / length
    d1 = np.linalg.norm(samples - v0, axis=1)
    d2 = np.linalg.norm(samples - v1, axis=1)
    dproj = np.dot(samples - v0, dir)
    proj = dproj[:,np.newaxis]*dir + v0
    d0 = np.linalg.norm(samples - proj, axis=1)
    d0[dproj < 0] = d1[dproj < 0]
    d0[dproj > length] = d2[dproj > length]
    dist = np.min((d1,d2,d0),axis=0)
    return dist
    

def distance_to_face(samples, v0, v1, v2):
    n = np.cross(v1 - v0, v2 - v0)
    n = n / np.linalg.norm(n)
    dist0 = np.dot(samples -  v0, n)

    projt = samples - dist0[:,np.newaxis] * n

    distt = np.abs(dist0)

    a2 = np.dot(np.cross(v1 - v0, projt - v0), n)
    a1 = np.dot(np.cross(v2 - v1, projt - v1), n)
    a0 = np.dot(np.cross(v0 - v2, projt - v2), n)

    inside = (a2 >= 0) & (a1 >= 0) & (a0 >= 0)

    diste0 = distance_to_edge(samples, v1, v2)
    diste1 = distance_to_edge(samples, v2, v0)
    diste2 = distance_to_edge(samples, v0, v1)

    distv0 = np.linalg.norm(samples - v0, axis=1)
    distv1 = np.linalg.norm(samples - v1, axis=1)
    distv2 = np.linalg.norm(samples - v2, axis=1)

    distt[inside == False] = distv0[inside == False]

    return np.min((distt, diste0, diste1, diste2, distv0, distv1, distv2), axis = 0)

    
        
def semi_discrete_distance_to_proj(samples,vertices, edges, faces):
    
    distv = distance_to_vertices(samples, vertices)
    diste = np.empty([np.shape(samples)[0],np.shape(edges)[0]])
    distt = np.empty([np.shape(samples)[0],np.shape(faces)[0]])

    ie = 0
    for e in edges:
        v0 = vertices[e[0] - 1,:]
        v1 = vertices[e[1] - 1,:]
        diste[:,ie] = distance_to_edge(samples, v0, v1)
        ie = ie + 1
    
    it = 0
    for t in faces:
        v0 = vertices[t[0] - 1,:]
        v1 = vertices[t[1] - 1,:]
        v2 = vertices[t[2] - 1,:]
        distt[:,it] = distance_to_face(samples, v0, v1, v2)
        it = it + 1

    
    return np.min((distt.min(axis = 1), diste.min(axis = 1), distv), axis = 0)

def semi_discrete_chamfer(samples,vertices, edges, faces):
    d = semi_discrete_distance_to_proj(samples, vertices, edges, faces)
    return np.mean(np.square(d))

def semi_discrete_directed_hausdorff(samples,vertices, edges, faces):
    d = semi_discrete_distance_to_proj(samples, vertices, edges, faces)
    return np.max(d)


#d = distance_to_face(np.array([np.array([-1,-1,-1]),np.array([1,1,1])]),np.array([0,0,0]),np.array([0,2,0]),np.array([0,0,2]))
#print(d)
#samples = np.array([np.array([-1,-1,-1]),np.array([1,1,1]),np.array([0,0.5,0.5]),np.array([0,0,3]),np.array([0,1,0.2])])
#vertices = np.array([np.array([0,0,0]),np.array([0,2,0]),np.array([0,0,2])])
#edges = np.array([np.array([1,2]),np.array([2,3]),np.array([3,1])])
#faces = np.array([np.array([1,2,3])])
#
#d = semi_discrete_distance_to_proj(samples, vertices, edges, faces)
#print(d)
#compare_skeletons('Results/cvsk_coverage_skel7_clean.obj','Results/cvsk_coverage_skel7_crop1r.obj')
#compare_skeletons('Results/cvsk_coverage_skel7_clean.obj','Results/cvsk_coverage_skel7_clean.obj',verbose=True)