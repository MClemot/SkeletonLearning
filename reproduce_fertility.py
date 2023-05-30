import torch
import os
import sys
import time
from io3d import to_obj, from_obj
from build_neural_skeleton import build_neural_skeleton
from build_igr_neural_skeleton import build_igr_neural_skeleton
from build_siren_neural_skeleton import build_siren_neural_skeleton
from build_coverage_skeleton import build_coverage_skeleton
from display import display_sdfColor, display_grad, display_gradgrad
from scipy.spatial.transform import Rotation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#creating directories if they do not exist
if not os.path.exists('Results'):
    os.makedirs('Results')

if not os.path.exists('Slices'):
    os.makedirs('Slices')

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

objects = [["fertility_v0r"  , 0.04, 0.04, 0.04, 1000],
           ["fertility_v0.5r", 0.04, 0.1 , 0.2 , 1000],
           ["fertility_v1r"  , 0.06, 0.1 , 0.2 , 1000],
           ["fertility_v2r"  , 0.06, 0.1 , 0.2 , 1000]]

if len(sys.argv) > 1:
    for shape in objects:
        s = shape[0]
        delta = shape[1]
        deltarelu = shape[2]
        deltacov = shape[3]

        delta = deltarelu
        time_limit = shape[4]
        print("************ Processing Shape",s,"***************")
        filename = "Objects/fertility_noise/{}.obj".format(s)
        filename_normals = "Objects/fertility_noise/{}_normals.obj".format(s)

        print(filename)
    
        pc = from_obj(filename)
        nc = from_obj(filename_normals)
        
        M = 1
        vertices, edges, triangles, net, skpts, upts = build_neural_skeleton(pc, nc, tv = True, activation="Sine", npl=npl, dep=dep, hints=10000, delta=delta, time_limit=time_limit)
        torch.save(net, "Networks/net_{}_{}_{}_{}.net".format(npl, dep, activation+("" if tv else "NoTV"), s))
        to_obj(vertices*M, "Results/cvsk_{}_{}.obj".format("Sine", s), lines=edges, tri=triangles)
        to_obj(upts*M, "Results/unif_points_{}_{}.obj".format("Sine", s))
        to_obj(skpts*M, "Results/skeletal_points_{}_{}.obj".format("Sine", s))

        vertices, edges, triangles, net, skpts, upts = build_igr_neural_skeleton(pc, nc, npl=npl, dep=dep, delta=delta, time_limit=time_limit)
        torch.save(net, "Networks/net_{}_{}_{}_{}.net".format(npl, dep, "igr", s))
        to_obj(vertices*M, "Results/cvsk_{}_{}.obj".format("igr", s), lines=edges, tri=triangles)
        to_obj(upts*M, "Results/unif_points_{}_{}.obj".format("igr", s))
        to_obj(skpts*M, "Results/skeletal_points_{}_{}.obj".format("igr", s))

        vertices, edges, triangles, net, skpts, upts = build_siren_neural_skeleton(pc, nc, npl=npl, dep=dep, delta=delta, time_limit=time_limit)
        torch.save(net, "Networks/net_{}_{}_{}_{}.net".format(npl, dep, "siren", s))
        to_obj(vertices*M, "Results/cvsk_{}_{}.obj".format("siren", s), lines=edges, tri=triangles)
        to_obj(upts*M, "Results/unif_points_{}_{}.obj".format("siren", s))
        to_obj(skpts*M, "Results/skeletal_points_{}_{}.obj".format("siren", s))
        
        vertices_cov, edges_cov, triangles_cov, covcandidates = build_coverage_skeleton(pc, nc, delta = deltacov, npts = 10000, time_limit=time_limit)
        to_obj(covcandidates*M, "Results/cov_cand_{}.obj".format(s))
        to_obj(vertices_cov*M, "Results/cvsk_{}_{}.obj".format("coverage", s), lines=edges_cov, tri=triangles_cov)
    
        
    print("skeletons computed in ", '{:.2f}'.format(time.time()-tskel),"s.")


for shape in objects:
    s = shape[0]

    slice = ['z', 0]

    r = Rotation.from_euler("xyz",[180, 135, 0],degrees=True)
    arr = 1.5*r.as_matrix()

    rotmatrix = torch.Tensor(arr)

    net = torch.load("Networks/net_{}_{}_{}_{}.net".format(npl, dep, "Sine", s))
    display_sdfColor(net, 400, slice[1], axis=slice[0],filename='Slices/slice_{}_{}_{}_{}_{}{}.png'.format(npl, dep, "Sine", s, slice[0], slice[1]), matrix = rotmatrix)
    display_grad(net, 400, slice[1], axis=slice[0], filename='Slices/slicegrad_{}_{}_{}_{}_{}{}.png'.format(npl, dep, "Sine", s, slice[0], slice[1]), matrix = rotmatrix)
    display_gradgrad(net, 200, slice[1], axis=slice[0], filename='Slices/slicegradgrad_{}_{}_{}_{}_{}{}.png'.format(npl, dep, "Sine", s, slice[0], slice[1]), matrix = rotmatrix)

    net1 = torch.load("Networks/net_{}_{}_{}_{}.net".format(npl, dep, "igr", s))
    display_sdfColor(net1, 400, slice[1], axis=slice[0], filename='Slices/slice_{}_{}_{}_{}_{}{}.png'.format(npl, dep, "igr", s, slice[0], slice[1]), matrix = rotmatrix)
    display_grad(net1, 400, slice[1], axis=slice[0], filename='Slices/slicegrad_{}_{}_{}_{}_{}{}.png'.format(npl, dep, "igr", s, slice[0], slice[1]), matrix = rotmatrix)
    display_gradgrad(net1, 200, slice[1], axis=slice[0], filename='Slices/slicegradgrad_{}_{}_{}_{}_{}{}.png'.format(npl, dep, "igr", s, slice[0], slice[1]), matrix = rotmatrix)

    net2 = torch.load("Networks/net_{}_{}_{}_{}.net".format(npl, dep, "siren", s))
    display_sdfColor(net2, 400, slice[1], axis=slice[0], filename='Slices/slice_{}_{}_{}_{}_{}{}.png'.format(npl, dep, "siren", s, slice[0], slice[1]), matrix = rotmatrix)
    display_grad(net2, 400, slice[1], axis=slice[0], filename='Slices/slicegrad_{}_{}_{}_{}_{}{}.png'.format(npl, dep, "siren", s, slice[0], slice[1]), matrix = rotmatrix)
    display_gradgrad(net2, 200, slice[1], axis=slice[0], filename='Slices/slicegradgrad_{}_{}_{}_{}_{}{}.png'.format(npl, dep, "siren", s, slice[0], slice[1]), matrix = rotmatrix)
