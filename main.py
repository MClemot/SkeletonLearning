import torch
import os
import sys
import time
import numpy as np
from io3d import to_obj, from_obj, from_xyz_normals
from geometry import sample_mesh
from build_neural_skeleton import build_neural_skeleton

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
delta = 0.03

tskel = time.time()

str_input = sys.argv[1]
str_output = sys.argv[2]

if str_input.endswith('.obj'):
    pc, nc, M = sample_mesh(str_input, 100000)
elif str_input.endswith('.xyz'):
    pc, nc = from_xyz_normals(str_input)
    M = np.max(pc)
    
vertices, edges, triangles, net, skpts, upts = build_neural_skeleton(pc, nc, tv=tv, activation=activation, npl=npl, dep=dep, hints=10000, delta=delta, time_limit=100)

to_obj(vertices * M, str_output, lines=edges, tri=triangles)
        
print("Skeleton computed in", '{:.2f}'.format(time.time()-tskel),"s.")
