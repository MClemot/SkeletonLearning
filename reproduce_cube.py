import torch
import os
import sys
import time
from geometry import sample_mesh
from io3d import to_obj, from_xyz_normals
from build_neural_skeleton import build_neural_skeleton
from build_siren_neural_skeleton import build_siren_neural_skeleton
from build_igr_neural_skeleton import build_igr_neural_skeleton
from display import display_sdfColor, display_grad, display_gradgrad

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#creating directories if they do not exist
if not os.path.exists('Results'):
    os.makedirs('Results')

if not os.path.exists('Slices'):
    os.makedirs('Slices')

if not os.path.exists('Networks'):
    os.makedirs('Networks')

# =============================================================================
# pretraining
# =============================================================================

# net = pretrain(dim_hidden=64, num_layers=6, skip=[], lr=2e-5, batch_size=25000, epochs=5000, activation="ReLU")
# torch.save(net, "Pretrained/pretrained_{}_{}_{}.net".format(64, 6, "ReLU"))

# =============================================================================
# neural SDF optimization
# =============================================================================

activation = "Sine"
tv = True
npl, dep = 64,6
mindist = 0.02

shapes = [["cube.xyz", 0.02, 0.02, 0.04, 0.02, 200]]
tabledata = []

tskel = time.time()

#train the networks if "0" is passed as an argument
if len(sys.argv) > 1:
    if int(sys.argv[1]) == 0:
        for shape in shapes:
            s = os.path.splitext(shape[0])[0]
            print(s)
            delta = shape[1]
            deltasiren = shape[2]
            deltarelu = shape[3]
            deltacov = shape[4]
            time_limit = shape[5]
            nhints = 1000
            print("\n\n************ Processing Shape",s,"***************")
            filename = "Objects/{}".format(shape[0])

            if filename.endswith('.xyz'):
                pc, nc = from_xyz_normals(filename)
                idx = torch.randint(high=pc.shape[0], size=(100000,))
                pc = pc[idx]
                nc = nc[idx]
                M = 1
            elif filename.endswith(('.obj')):
                pc, nc, M = sample_mesh(filename, 100000)

            print("****** Shape {}, Sine with TV (Ours) ************".format(s))

            netpath = None #netpath = "Networks/net_{}_{}_{}_{}.net".format(npl, dep, "Sine", s)
            vertices, edges, triangles, net, skpts, upts = build_neural_skeleton(pc, nc, tv = True,
                                                                                activation="Sine", npl=64, dep=6,
                                                                                hints=nhints, delta=delta,
                                                                                time_limit=time_limit, trainednet=netpath, scaleshape=False)
            torch.save(net, "Networks/net_{}_{}_{}_{}.net".format(npl, dep, "Sine", s))
            to_obj(vertices*M, "Results/cvsk_{}_{}.obj".format("Sine", s), lines=edges, tri=triangles)
            to_obj(upts*M, "Results/unif_points_{}_{}.obj".format("Sine", s))
            to_obj(skpts*M, "Results/skeletal_points_{}_{}.obj".format("Sine", s))

            print("****** Shape {}, Sine without TV ************".format(s))

            netpath = None #netpath = None#"Networks/net_{}_{}_{}_{}.net".format(npl, dep, "SineNoTV", s)
            vertices, edges, triangles, net1, skpts, upts = build_neural_skeleton(pc, nc, tv = False,
                                                                                activation="Sine", npl=64, dep=6,
                                                                                hints=nhints, delta=deltasiren,
                                                                                time_limit=time_limit, trainednet=netpath, scaleshape=False)
            torch.save(net1, "Networks/net_{}_{}_{}_{}.net".format(npl, dep, "SineNoTV", s))
            to_obj(vertices*M, "Results/cvsk_{}_{}.obj".format("SineNoTV", s), lines=edges, tri=triangles)
            to_obj(upts*M, "Results/unif_points_{}_{}.obj".format("SineNoTV", s))
            to_obj(skpts*M, "Results/skeletal_points_{}_{}.obj".format("SineNoTV", s))


            print("****** Shape {}, ReLU without TV ************".format(s))

            netpath = None #netpath = None #"Networks/net_{}_{}_{}_{}.net".format(npl, dep, "ReLU", s)
            vertices, edges, triangles, net2, skpts, upts = build_neural_skeleton(pc, nc, tv = False,
                                                                                activation="ReLU", npl=64, dep=6,
                                                                                hints=nhints, delta = deltarelu,
                                                                                time_limit=time_limit, trainednet=netpath, scaleshape=False)
            torch.save(net2, "Networks/net_{}_{}_{}_{}.net".format(npl, dep, "ReLU", s))
            to_obj(vertices*M, "Results/cvsk_{}_{}.obj".format("ReLU", s), lines=edges, tri=triangles)
            to_obj(upts*M, "Results/unif_points_{}_{}.obj".format("ReLU", s))
            to_obj(skpts*M, "Results/skeletal_points_{}_{}.obj".format("ReLU", s))


            print("****** Shape {}, SIREN ************".format(s))

            netpath = None #"Networks/net_{}_{}_{}_{}.net".format(npl, dep, "siren", s)
            vertices, edges, triangles, net1, skpts, upts = build_siren_neural_skeleton(pc, nc, npl=64, dep=6, delta=deltasiren,
                                                                                time_limit=time_limit, trainednet=netpath, scaleshape=False, plot=True)
            torch.save(net1, "Networks/net_{}_{}_{}_{}.net".format(npl, dep, "siren", s))
            to_obj(vertices*M, "Results/cvsk_{}_{}.obj".format("siren", s), lines=edges, tri=triangles)
            to_obj(upts*M, "Results/unif_points_{}_{}.obj".format("siren", s))
            to_obj(skpts*M, "Results/skeletal_points_{}_{}.obj".format("siren", s))


            print("****** Shape {}, IGR ************".format(s))

            netpath = None #"Networks/net_{}_{}_{}_{}.net".format(npl, dep, "igr", s)
            vertices, edges, triangles, net2, skpts, upts = build_igr_neural_skeleton(pc, nc, npl=64, dep=6,
                                                                                delta = deltarelu,
                                                                                time_limit=time_limit, trainednet=netpath, scaleshape=False)
            torch.save(net2, "Networks/net_{}_{}_{}_{}.net".format(npl, dep, "igr", s))
            to_obj(vertices*M, "Results/cvsk_{}_{}.obj".format("igr", s), lines=edges, tri=triangles)
            to_obj(upts*M, "Results/unif_points_{}_{}.obj".format("igr", s))
            to_obj(skpts*M, "Results/skeletal_points_{}_{}.obj".format("igr", s))


print("skeletons computed in ", '{:.2f}'.format(time.time()-tskel),"s.")


for shape in shapes:
    s = os.path.splitext(shape[0])[0]
    delta = shape[1]
    deltarelu = shape[2]
    deltacov = shape[3]

    slice = ['z', 0]

    net = torch.load("Networks/net_{}_{}_{}_{}.net".format(npl, dep, "Sine", s))
    display_sdfColor(net, 400, slice[1], axis=slice[0],filename='Slices/slice_{}_{}_{}_{}_{}{}.png'.format(npl, dep, "Sine", s, slice[0], slice[1]))
    display_grad(net, 400, slice[1], axis=slice[0], filename='Slices/slicegrad_{}_{}_{}_{}_{}{}.png'.format(npl, dep, "Sine", s, slice[0], slice[1]))
    display_gradgrad(net, 200, slice[1], axis=slice[0], filename='Slices/slicegradgrad_{}_{}_{}_{}_{}{}.png'.format(npl, dep, "Sine", s, slice[0], slice[1]))

    net1 = torch.load("Networks/net_{}_{}_{}_{}.net".format(npl, dep, "siren", s))
    display_sdfColor(net1, 400, slice[1], axis=slice[0], filename='Slices/slice_{}_{}_{}_{}_{}{}.png'.format(npl, dep, "siren", s, slice[0], slice[1]))
    display_grad(net1, 400, slice[1], axis=slice[0], filename='Slices/slicegrad_{}_{}_{}_{}_{}{}.png'.format(npl, dep, "siren", s, slice[0], slice[1]))
    display_gradgrad(net1, 200, slice[1], axis=slice[0], filename='Slices/slicegradgrad_{}_{}_{}_{}_{}{}.png'.format(npl, dep, "siren", s, slice[0], slice[1]))

    net2 = torch.load("Networks/net_{}_{}_{}_{}.net".format(npl, dep, "igr", s))
    display_sdfColor(net2, 400, slice[1], axis=slice[0], filename='Slices/slice_{}_{}_{}_{}_{}{}.png'.format(npl, dep, "igr", s, slice[0], slice[1]))
    display_grad(net2, 400, slice[1], axis=slice[0], filename='Slices/slicegrad_{}_{}_{}_{}_{}{}.png'.format(npl, dep, "igr", s, slice[0], slice[1]))
    display_gradgrad(net2, 200, slice[1], axis=slice[0], filename='Slices/slicegradgrad_{}_{}_{}_{}_{}{}.png'.format(npl, dep, "igr", s, slice[0], slice[1]))
