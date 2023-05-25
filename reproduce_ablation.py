import numpy as np
import torch
import os
import sys
import time

from io3d import to_obj, from_xyz_normals
from build_neural_skeleton import build_neural_skeleton
from distances import compare_skeletal_points_to_gtskel
from tabulate import tabulate


mindist = 0.01

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#creating Ablation dir if it does not exist
if not os.path.exists('Ablation'):
    os.makedirs('Ablation')

#shapes = [["skel7_clean", 0.03, 0.04, 0.03]]
shapes = [["skel7_clean", 0.03, 0.04, 0.03], ["skel7_crop1r",0.035, 0.04, 0.03], ["skel7_crop2r",0.03, 0.04, 0.03],
          ["skel7_crop3r",0.03, 0.04, 0.03], ["skel7_crop4r", 0.03, 0.04, 0.03], ["skel7_sub25r",0.03, 0.04, 0.03],
          ["skel7_sub50r",0.03, 0.04, 0.03], ["skel7_var0.05r",0.03, 0.04, 0.03], ["skel7_var0.1r",0.03, 0.04, 0.03], ["skel7_var0.5r", 0.035, 0.04, 0.05]]
gt = "Benchmark/skel7/skel7_groundtruth.obj"
tabledata = []

activation = "Sine"
tv = True
npl, dep = 64,6
time_limit=360
tskel = time.time()
nbhints=5000

#train the networks if "0" is passed as an argument
if len(sys.argv) > 1:
    if int(sys.argv[1]) == 0:
        for shape in shapes:
            s = shape[0]
            delta = shape[1]

            print("************ Processing Shape ",s,"***************")
            filename = "Benchmark/skel7/{}.xyz".format(s)
            pc, nc = from_xyz_normals(filename)
            
            vertices, edges, triangles, net, skpts, upts = build_neural_skeleton(pc, nc, tv=True, activation="Sine", npl=64, dep=6, hints=nbhints, delta=delta, time_limit=time_limit)
            torch.save(net, "Networks/net_{}_{}_{}_{}.net".format(npl, dep, "Sine", s))
            to_obj(vertices, "Results_benchmark1/cvsk_{}_{}.obj".format("Sine", s), lines=edges, tri=triangles)
            to_obj(upts, "Results_benchmark1/unif_points_{}_{}.obj".format("Sine", s))
            to_obj(skpts, "Results_benchmark1/skeletal_points_{}_{}.obj".format("Sine", s))

            #print("no isopoints")
            vertices, edges, triangles, net, skpts, upts = build_neural_skeleton(pc, nc, tv=True, activation="Sine", npl=64, dep=6, hints=nbhints, resampling=False, delta=shape[2], time_limit=time_limit)
            torch.save(net, "Networks/net_{}_{}_{}_{}.net".format(npl, dep, 'Sine_noisopts', s))
            to_obj(vertices, "Results_benchmark1/cvsk_{}_{}.obj".format("Sine_noisopts", s), lines=edges, tri=triangles)
            to_obj(upts, "Results_benchmark1/unif_points_{}_{}.obj".format("Sine_noisopts", s))
            to_obj(skpts, "Results_benchmark1/skeletal_points_{}_{}.obj".format("Sine_noisopts", s))

            vertices, edges, triangles, net, skpts, upts = build_neural_skeleton(pc, nc, tv=False, activation="Sine", npl=64, dep=6, hints=nbhints, delta = delta, time_limit=time_limit)
            torch.save(net, "Networks/net_{}_{}_{}_{}.net".format(npl, dep, "SineNoTV", s))
            to_obj(vertices, "Results_benchmark1/cvsk_{}_{}.obj".format("SineNoTV", s), lines=edges, tri=triangles)
            to_obj(upts, "Results_benchmark1/unif_points_{}_{}.obj".format("SineNoTV", s))
            to_obj(skpts, "Results_benchmark1/skeletal_points_{}_{}.obj".format("SineNoTV", s))

            vertices, edges, triangles, net, skpts, upts = build_neural_skeleton(pc, nc, tv=False, activation="ReLU", npl=64, dep=6, hints=nbhints, delta = delta, time_limit=time_limit)
            torch.save(net, "Networks/net_{}_{}_{}_{}.net".format(npl, dep, "ReLU", s))
            to_obj(vertices, "Results_benchmark1/cvsk_{}_{}.obj".format("ReLU", s), lines=edges, tri=triangles)
            to_obj(upts, "Results_benchmark1/unif_points_{}_{}.obj".format("ReLU", s))
            to_obj(skpts, "Results_benchmark1/skeletal_points_{}_{}.obj".format("ReLU", s))


            vertices, edges, triangles, net, skpts, upts = build_neural_skeleton(pc, nc, tv=False, activation="SoftPlus", npl=64, dep=6,
                                                                                 hints=nbhints, delta = shape[3], time_limit=time_limit)#,
                                                                                 #trainednet="Networks/net_{}_{}_{}_{}.net".format(npl, dep, "SoftPlusNoTV", s))
            torch.save(net, "Networks/net_{}_{}_{}_{}.net".format(npl, dep, "SoftPlusNoTV", s))
            to_obj(vertices, "Results_benchmark1/cvsk_{}_{}.obj".format("SoftPlusNoTV", s), lines=edges, tri=triangles)
            to_obj(upts, "Results_benchmark1/unif_points_{}_{}.obj".format("SoftPlusNoTV", s))
            to_obj(skpts, "Results_benchmark1/skeletal_points_{}_{}.obj".format("SoftPlusNoTV", s))


            vertices, edges, triangles, net, skpts, upts = build_neural_skeleton(pc, nc, tv=True, activation="SoftPlus", npl=64, dep=6, hints=nbhints,
                                                                                 delta = shape[3], time_limit=time_limit,
                                                                                 trainednet="Networks/net_{}_{}_{}_{}.net".format(npl, dep, "SoftPlus", s))
            torch.save(net, "Networks/net_{}_{}_{}_{}.net".format(npl, dep, "SoftPlus", s))
            to_obj(vertices, "Results_benchmark1/cvsk_{}_{}.obj".format("SoftPlus", s), lines=edges, tri=triangles)
            to_obj(upts, "Results_benchmark1/unif_points_{}_{}.obj".format("SoftPlus", s))
            to_obj(skpts, "Results_benchmark1/skeletal_points_{}_{}.obj".format("SoftPlus", s))
         
            vertices, edges, triangles, net, skpts, upts = build_neural_skeleton(pc, nc, tv = True, activation="Sine", npl=64, dep=6, hints=0, delta=shape[3], time_limit=time_limit)
            torch.save(net, "Networks/net_{}_{}_{}_{}.net".format(npl, dep, 'Sine_nohint', s))
            to_obj(vertices, "Results_benchmark1/cvsk_{}_{}.obj".format("Sine_nohint", s), lines=edges, tri=triangles)
            to_obj(upts, "Results_benchmark1/unif_points_{}_{}.obj".format("Sine_nohint", s))
            to_obj(skpts, "Results_benchmark1/skeletal_points_{}_{}.obj".format("Sine_nohint", s))


print("skeletons computed in ", '{:.2f}'.format(time.time()-tskel),"s.")


tdist = time.time()

#compute distances
print("Computing distances to groundtruth")
tabledata = []
gt = "Benchmark/skel7/skel7_groundtruth.obj"
for shape in shapes:
    s = shape[0]
    print("************ {} ***************".format(s))
    filename = "Benchmark/skel7/{}.xyz".format(s)
    
    print("Ours")
    cdf, hdf = compare_skeletal_points_to_gtskel(gt, "Results_benchmark1/skeletal_points_{}_{}.obj".format("Sine",s), mindist = mindist)

    print("SineNoTV")
    cds, hds = compare_skeletal_points_to_gtskel(gt, "Results_benchmark1/skeletal_points_SineNoTV_{}.obj".format(s), mindist=mindist)

    print("ReLU")
    cdr, hdr = compare_skeletal_points_to_gtskel(gt, "Results_benchmark1/skeletal_points_ReLU_{}.obj".format(s), mindist=mindist)

    print("SoftPlus No TV")
    cdspntv, hdspntv = compare_skeletal_points_to_gtskel(gt, "Results_benchmark1/skeletal_points_SoftPlusNoTV_{}.obj".format(s), mindist=mindist)

    print("SoftPlus")
    cdsp, hdsp = compare_skeletal_points_to_gtskel(gt, "Results_benchmark1/skeletal_points_SoftPlus_{}.obj".format(s), mindist=mindist)
    
    print("No hint")
    cdnh, hdnh = compare_skeletal_points_to_gtskel(gt, "Results_benchmark1/skeletal_points_Sine_nohint_{}.obj".format(s), mindist=mindist)

    print("No isopoints")
    cdniso, hdniso = compare_skeletal_points_to_gtskel(gt, "Results_benchmark1/skeletal_points_Sine_noisopts_{}.obj".format(s), mindist = mindist)

    tabledata.append([ s, "{0:.3g}/{1:.3g}".format(cdf,hdf), "{0:.3g}/{1:.3g}".format(cds,hds),
                          "{0:.3g}/{1:.3g}".format(cdr,hdr), "{0:.3g}/{1:.3g}".format(cdsp,hdsp), "{0:.3g}/{1:.3g}".format(cdspntv,hdspntv),
                          "{0:.3g}/{1:.3g}".format(cdnh,hdnh),
                          "{0:.3g}/{1:.3g}".format(cdniso,hdniso)])

print("distances computed in ", '{:.2f}'.format(time.time()-tdist),"s.")

    
col_names=["Shape", "Full Method", "No TV", "ReLU", "SoftPlus", "SoftPlus, NoTV", "No hint", "No isopoints"]

print(tabulate(tabledata, headers=col_names, tablefmt="fancy_grid"))

with open('table_ablation.tex', 'w') as f: f.write(tabulate(tabledata, headers=col_names, tablefmt="latex"))

