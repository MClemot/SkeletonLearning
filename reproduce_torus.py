
import numpy as np
import torch
import os
import sys
import time
from io3d import to_obj, from_obj
from build_neural_skeleton import build_neural_skeleton
from build_coverage_skeleton import build_coverage_skeleton
from distances import compare_skeletal_points_to_gtpts
from tabulate import tabulate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#creating directories if they do not exist
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
mindist = 0.01

shapes = [["dtore_noise0.0"  , 0.02, 0.02, 0.02],
          ["dtore_noise0.003", 0.02, 0.02, 0.02],
          ["dtore_noise0.005", 0.02, 0.02, 0.02],
          ["dtore_noise0.01" , 0.02, 0.02, 0.02],
          ["dtore_noise0.03" , 0.02, 0.02, 0.02],
          ["dtore_trunc"     , 0.02, 0.02, 0.02],
          ["dtore_trunc2"    , 0.02, 0.02, 0.02],
          ["dtore_trunc3"    , 0.02, 0.02, 0.02],
          ["dtore_trunc4"    , 0.02, 0.02, 0.02]]
tabledata = []

tskel = time.time()
time_limit=1000


#train the networks if "0" is passed as an argument
if len(sys.argv) > 1:
    if int(sys.argv[1]) == 0:
        for shape in shapes:
            s = shape[0]
            delta = shape[1]
            deltarelu = shape[2]
            deltacov = shape[3]
            print("************ Processing Shape ",s,"***************")
            filename = "Objects/Tore/{}.obj".format(s)
            normalfilename = "Objects/Tore/{}_normals.obj".format(s)

            pc = from_obj(filename)
            nc = from_obj(normalfilename)
            M=1

            vertices, edges, triangles, net, skpts, upts = build_neural_skeleton(pc, nc, tv = True, activation="SoftPlus", npl=64, dep=6, hints=1000, delta=delta, time_limit=time_limit)
            torch.save(net, "Networks/net_{}_{}_{}_{}.net".format(npl, dep, "SoftPlus", s))
            to_obj(vertices*M, "Results/cvsk_{}_{}.obj".format("SoftPlus", s), lines=edges, tri=triangles)
            to_obj(upts*M, "Results/unif_points_{}_{}.obj".format("SoftPlus", s))
            to_obj(skpts*M, "Results/skeletal_points_{}_{}.obj".format("SoftPlus", s))

            vertices, edges, triangles, net, skpts, upts = build_neural_skeleton(pc, nc, tv = False, activation="SoftPlus", npl=64, dep=6, hints=10000, delta=delta, time_limit=time_limit)
            torch.save(net, "Networks/net_{}_{}_{}_{}.net".format(npl, dep, "SoftPlusNoTV", s))
            to_obj(vertices*M, "Results/cvsk_{}_{}.obj".format("SoftPlusNoTV", s), lines=edges, tri=triangles)
            to_obj(upts*M, "Results/unif_points_{}_{}.obj".format("SoftPlusNoTV", s))
            to_obj(skpts*M, "Results/skeletal_points_{}_{}.obj".format("SoftPlusNoTV", s))

            vertices, edges, triangles, net, skpts, upts = build_neural_skeleton(pc, nc, tv = True, activation="Sine", npl=64, dep=6, hints=10000, delta=delta, time_limit=time_limit)
            torch.save(net, "Networks/net_{}_{}_{}_{}.net".format(npl, dep, "Sine"+("" if tv else "NoTV"), s))
            to_obj(vertices*M, "Results/cvsk_{}_{}.obj".format("Sine"+("" if tv else "NoTV"), s), lines=edges, tri=triangles)
            to_obj(upts*M, "Results/unif_points_{}_{}.obj".format("Sine", s))
            to_obj(skpts*M, "Results/skeletal_points_{}_{}.obj".format("Sine", s))

            vertices_siren, edges_siren, triangles_siren, net_siren, skpts_siren, upts_siren = build_neural_skeleton(pc, nc, tv = False, activation="Sine", npl=64, dep=6, hints=10000, delta=delta, time_limit=time_limit)
            torch.save(net_siren, "Networks/net_{}_{}_{}_{}.net".format(npl, dep, "SineNoTV", s))
            to_obj(vertices_siren*M, "Results/cvsk_{}_{}.obj".format("SineNoTV", s), lines=edges_siren, tri=triangles_siren)
            to_obj(upts_siren*M, "Results/unif_points_{}_{}.obj".format("SineNoTV", s))
            to_obj(skpts_siren*M, "Results/skeletal_points_{}_{}.obj".format("SineNoTV", s))

            vertices_relu, edges_relu, triangles_relu, net_relu, skpts_relu, upts_relu = build_neural_skeleton(pc, nc, tv = False, activation="ReLU", npl=64, dep=6, hints=10000, delta=deltarelu, time_limit=time_limit)
            torch.save(net_relu, "Networks/net_{}_{}_{}_{}.net".format(npl, dep, "ReLU"+("" if tv else "NoTV"), s))
            to_obj(vertices_relu*M, "Results/cvsk_{}_{}.obj".format("ReLU"+("" if tv else "NoTV"), s), lines=edges_relu, tri=triangles_relu)
            to_obj(upts_relu*M, "Results/unif_points_{}_{}.obj".format("ReLU", s))
            to_obj(skpts_relu*M, "Results/skeletal_points_{}_{}.obj".format("ReLU", s))
    
            vertices_cov, edges_cov, triangles_cov, covcandidates = build_coverage_skeleton(pc, nc, delta = deltacov, npts = 10000, time_limit=time_limit)
            to_obj(covcandidates*M, "Results/cov_cand_{}.obj".format(s))
            to_obj(vertices_cov*M, "Results/cvsk_{}_{}.obj".format("coverage", s), lines=edges_cov, tri=triangles_cov)

            vertices, edges, triangles, net, skpts, upts = build_neural_skeleton(pc, nc, tv=True, activation="Sine", npl=64, dep=6, hints=10000, resampling=False, delta=shape[2], time_limit=time_limit)
            torch.save(net, "Networks/net_{}_{}_{}_{}.net".format(npl, dep, 'Sine_noisopts', s))
            to_obj(vertices, "Results/cvsk_{}_{}.obj".format("Sine_noisopts", s), lines=edges, tri=triangles)
            to_obj(upts, "Results/unif_points_{}_{}.obj".format("Sine_noisopts", s))
            to_obj(skpts, "Results/skeletal_points_{}_{}.obj".format("Sine_noisopts", s))

            vertices, edges, triangles, net, skpts, upts = build_neural_skeleton(pc, nc, tv = True, activation="Sine", npl=64, dep=6, hints=0, delta=shape[3], time_limit=time_limit)
            torch.save(net, "Networks/net_{}_{}_{}_{}.net".format(npl, dep, 'Sine_nohint', s))
            to_obj(vertices, "Results/cvsk_{}_{}.obj".format("Sine_nohint", s), lines=edges, tri=triangles)
            to_obj(upts, "Results/unif_points_{}_{}.obj".format("Sine_nohint", s))
            to_obj(skpts, "Results/skeletal_points_{}_{}.obj".format("Sine_nohint", s))
            
print("skeletons computed in ", '{:.2f}'.format(time.time()-tskel),"s.")


print("Computing distances to groundtruth")

angles = np.arange(0, 2*np.pi, np.pi/100000)
size = np.shape(angles)[0]
gt = 0.7*np.concatenate((np.cos(angles[:,np.newaxis]),np.sin(angles[:,np.newaxis]),np.zeros((size,1))),1)

tdist = time.time()
for shape in shapes:
    s = shape[0]
    print("****************{}****************".format(s))
    print("Ours")
    cdn, hdn = compare_skeletal_points_to_gtpts(gt, "Results/skeletal_points_{}_{}.obj".format("Sine"+("" if tv else "NoTV"),s), mindist = mindist)
    print("ReLU")
    cdrelu, hdrelu = compare_skeletal_points_to_gtpts(gt, "Results/skeletal_points_{}_{}.obj".format("ReLU",s), mindist = mindist)
    print("Coverage")
    cdc, hdc = compare_skeletal_points_to_gtpts(gt, "Results/cov_cand_{}.obj".format(s), mindist=mindist)
    print("Siren")
    cdsiren, hdsiren = compare_skeletal_points_to_gtpts(gt, "Results/skeletal_points_{}_{}.obj".format("SineNoTV", s), mindist=mindist)
    print("SoftPlus")
    cdsoftplusTV, hdsoftplusTV = compare_skeletal_points_to_gtpts(gt, "Results/skeletal_points_{}_{}.obj".format("SoftPlus", s), mindist=mindist)
    print("SoftPlusNoTV")
    cdsoftplus, hdsoftplus = compare_skeletal_points_to_gtpts(gt, "Results/skeletal_points_{}_{}.obj".format("SoftPlusNoTV", s), mindist=mindist)
    print("no isopoints")
    cdisopts, hdisopts = compare_skeletal_points_to_gtpts(gt, "Results/skeletal_points_{}_{}.obj".format("Sine_noisopts", s), mindist=mindist)
    print("no hints")
    cdhint, hdhint = compare_skeletal_points_to_gtpts(gt, "Results/skeletal_points_{}_{}.obj".format("Sine_nohint", s), mindist=mindist)

    tabledata.append([ s, "{0:.3g}/{1:.3g}".format(cdn,hdn), "{0:.3g}/{1:.3g}".format(cdsiren,hdsiren),
                          "{0:.3g}/{1:.3g}".format(cdrelu,hdrelu), "{0:.3g}/{1:.3g}".format(cdsoftplusTV,hdsoftplusTV),
                          "{0:.3g}/{1:.3g}".format(cdsoftplus,hdsoftplus),"{0:.3g}/{1:.3g}".format(cdc,hdc),
                          "{0:.3g}/{1:.3g}".format(cdisopts,hdisopts), "{0:.3g}/{1:.3g}".format(cdhint,hdhint)])

print("distances computed in ", '{:.2f}'.format(time.time()-tdist),"s.")

col_names=["Shape", "Ours", "No TV", "ReLU, No TV", "SoftPlus", "SoftPlus, No TV", "Cov Axis", "No isopoints", "No hints"]

print(tabulate(tabledata, headers=col_names, tablefmt="fancy_grid"))
with open('table_torus.tex', 'w') as f: f.write(tabulate(tabledata, headers=col_names, tablefmt="latex"))