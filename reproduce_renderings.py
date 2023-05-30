import torch
import os
import sys
import time
from io3d import to_obj, from_obj
from geometry import sample_mesh
from build_neural_skeleton import build_neural_skeleton
from render import render

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#creating Results dir if it does not exist
if not os.path.exists('Results'):
    os.makedirs('Results')
    
if not os.path.exists('Networks'):
    os.makedirs('Networks')

# =============================================================================
# neural SDF optimization
# =============================================================================

activation = "Sine"
tv = True
npl, dep = 64,6
mindist = 0.02

tskel = time.time()

objects = [["birdcage", 0.03, 100],
           ["bitore", 0.03, 100],
           ["buckminsterfullerene", 0.03, 200],
           ["chair", 0.03, 200],
           ["dino", 0.03, 100],
           ["fertility", 0.04, 100],
           ["hand", 0.03, 100],
           ["helice", 0.03, 100],
           ["hilbert", 0.04, 100],
           ["lamp", 0.025, 100],
           ["metatron", 0.03, 100],
           ["pillowbox", 0.03, 100],
           ["protein", 0.03, 100],
           ["spot", 0.03, 100],
           ["zilla", 0.035, 100]]

for shape in objects:
    s = shape[0]
    delta = shape[1]
    
    filename = "Objects/{}.obj".format(s)
    filename_normals = "Objects/{}_normals.obj".format(s)
    
    netpath = "Networks/net_{}_{}_{}_{}.net".format(npl, dep, "Sine"+("" if tv else "NoTV"), s)
    skpath = "Networks/cvsk_{}_{}.obj".format("Sine"+("" if tv else "NoTV"), s)
    if len(sys.argv) == 1 or s in sys.argv[1:]:
        print("************ Processing Shape",s,"***************")
        if os.path.exists(filename_normals):
            pc = from_obj(filename)
            nc = from_obj(filename_normals)
            M = 1
        else:
            pc, nc, M = sample_mesh(filename, 100000)
        vertices, edges, triangles, net, skpts, upts = build_neural_skeleton(pc, nc, tv = tv, activation="Sine", npl=64, dep=6, hints=10000, delta=delta, time_limit=shape[2])#, trainednet=netpath)
        torch.save(net, "Networks/net_{}_{}_{}_{}.net".format(npl, dep, "Sine"+("" if tv else "NoTV"), s))

        to_obj(vertices*M, "Results/cvsk_{}_{}.obj".format("Sine"+("" if tv else "NoTV"), s), lines=edges, tri=triangles)
        to_obj(upts*M, "Results/unif_points_{}_{}.obj".format("Sine"+("" if tv else "NoTV"), s))
        to_obj(skpts*M, "Results/skeletal_points_{}_{}.obj".format("Sine"+("" if tv else "NoTV"), s))
        
        render(s)
        
print("skeletons computed in ", '{:.2f}'.format(time.time()-tskel),"s.")
