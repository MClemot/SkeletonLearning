import numpy as np
import torch
import os
import sys
import time
from io3d import to_obj, from_obj
from geometry import sample_mesh
from build_neural_skeleton import build_neural_skeleton


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#creating Results dir if it does not exist
if not os.path.exists('Results'):
    os.makedirs('Results')
if not os.path.exists('ResultsObj'):
    os.makedirs('ResultsObj')
if not os.path.exists('NetworksObj'):
    os.makedirs('NetworksObj')

# =============================================================================
# pretraining
# =============================================================================

# net = pretrain(dim_hidden=64, num_layers=6, skip=[], lr=2e-5, batch_size=25000, epochs=5000, activation="ReLU")
# torch.save(net, "Pretrained/pretrained_{}_{}_{}.net".format(64, 6, "ReLU"))

# =============================================================================
# neural SDF optimization
# =============================================================================

activation = "Sine"
tv = False
npl, dep = 64,6
mindist = 0.02


tskel = time.time()

objects = [#["bimba", 0.04, 1000],
           ["birdcage", 0.03, 1000],
           ["bitore", 0.03, 1000],
           ["buckminsterfullerene", 0.03, 1000],
           #["bunny", 0.04, 1000],
           #["dino", 0.03, 1000],
           #["dragon", 0.04, 1000],
           ["fertility", 0.04, 1000],
           #["guitar", 0.03, 1000],
           #["hand", 0.03, 1000],
           ["hand2", 0.03, 1000],
           #["helice", 0.03, 1000],
           ["hilbert", 0.04, 1000],
           ["metatron", 0.03, 1000],
           #["pillowbox", 0.03, 1000],
           #["protein", 0.03, 1000],
           #["spot", 0.03, 1000],
           ["vase", 0.04, 1000],
           #["yoga2", 0.08, 1000],
           ["zilla", 0.035, 1000],
           ["lamp", 0.025, 1000],
           ]

for shape in objects:
    s = shape[0]
    delta = shape[1]
    print("************ Processing Shape",s,"***************")
    filename = "Objects/{}.obj".format(s)
    filename_normals = "Objects/{}_normals.obj".format(s)
    
    netpath = "NetworksObj/net_{}_{}_{}_{}.net".format(npl, dep, "Sine"+("" if tv else "NoTV"), s)
    skpath = "ResultsObj/cvsk_{}_{}.obj".format("Sine"+("" if tv else "NoTV"), s)
    if len(sys.argv) > 1:
        if int(sys.argv[1]) == 1:
            pc, nc = None, None
            if not os.path.exists(skpath):
                vertices, edges, triangles, net, skpts, upts = build_neural_skeleton(pc, nc, tv = True, activation="Sine", npl=64, dep=6, hints=10000, delta=delta, trainednet=netpath)
            else:
                continue
        else:
            if os.path.exists(filename_normals):
                pc = from_obj(filename)
                nc = from_obj(filename_normals)
                M = 1
            else:
                pc, nc, M = sample_mesh(filename, 100000)
            vertices, edges, triangles, net, skpts, upts = build_neural_skeleton(pc, nc, tv = tv, activation="Sine", npl=64, dep=6, hints=10000, delta=delta, time_limit=shape[2])#, trainednet=netpath)
            torch.save(net, "NetworksObj/net_{}_{}_{}_{}.net".format(npl, dep, "Sine"+("" if tv else "NoTV"), s))

        to_obj(vertices*M, "ResultsObj/cvsk_{}_{}.obj".format("Sine"+("" if tv else "NoTV"), s), lines=edges, tri=triangles)
        to_obj(upts*M, "ResultsObj/unif_points_{}_{}.obj".format("Sine"+("" if tv else "NoTV"), s))
        to_obj(skpts*M, "ResultsObj/skeletal_points_{}_{}.obj".format("Sine"+("" if tv else "NoTV"), s))

        
        #vertices_cov, edges_cov, triangles_cov, covcandidates = build_coverage_skeleton(pc, nc, delta = delta, npts = 10000, time_limit=shape[2])
        #to_obj(covcandidates*M, "ResultsObj/cov_cand_{}.obj".format(s))
        #to_obj(vertices_cov*M, "ResultsObj/cvsk_{}_{}.obj".format("coverage", s), lines=edges_cov, tri=triangles_cov)
    
        
print("skeletons computed in ", '{:.2f}'.format(time.time()-tskel),"s.")
