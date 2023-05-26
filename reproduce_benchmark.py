import numpy as np
import torch
import os
import sys
import time
from io3d import to_obj, from_xyz_normals
from build_neural_skeleton import build_neural_skeleton
from build_siren_neural_skeleton import build_siren_neural_skeleton
from build_igr_neural_skeleton import build_igr_neural_skeleton
from build_coverage_skeleton import build_coverage_skeleton
from distances import compare_skeletal_points_to_gtskel
from tabulate import tabulate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#creating directories if they do not exist
if not os.path.exists('Results_benchmark'):
    os.makedirs('Results_benchmark')
    
if not os.path.exists('Networks'):
    os.makedirs('Networks')

# =============================================================================
# neural SDF optimization
# =============================================================================

activation = "Sine"
tv = True
npl, dep = 64,6
mindist = 0.01
res=[]

shapes = [["skel7_clean"   , 0.025, 0.06, 0.04, 0.03],
          ["skel7_crop1r"  , 0.025, 0.06, 0.04, 0.03],
          ["skel7_crop2r"  , 0.025, 0.06, 0.04, 0.03],
          ["skel7_crop3r"  , 0.025, 0.06, 0.04, 0.03],
          ["skel7_crop4r"  , 0.025, 0.06, 0.04, 0.03],
          ["skel7_sub25r"  , 0.025, 0.06, 0.04, 0.03],
          ["skel7_sub50r"  , 0.025, 0.06, 0.04, 0.03],
          ["skel7_var0.05r", 0.02 , 0.04, 0.04, 0.03],
          ["skel7_var0.1r" , 0.02 , 0.04, 0.04, 0.03],
          ["skel7_var0.5r" , 0.02 , 0.05, 0.05, 0.05],
          ["skel7_var1r"   , 0.02 , 0.05, 0.05, 0.05],
          ["skel7_var2r"   , 0.02 , 0.05, 0.05, 0.05]]
gt = "Objects/Benchmark/skel7_groundtruth.obj"
tabledata = []

tskel = time.time()
time_limit=10

tcdc = np.zeros(8)
thdc = np.zeros(8)
tcdn = np.zeros(8)
thdn = np.zeros(8)
tcdsiren = np.zeros(8)
thdsiren = np.zeros(8)
tcdrelu = np.zeros(8)
thdrelu = np.zeros(8)

for niter in range(0,1):

    #train the networks if "0" is passed as an argument
    if len(sys.argv) > 1:
        if int(sys.argv[1]) == 0:
            for shape in shapes:
                s = shape[0]
                delta = shape[1]
                deltasiren=shape[2]
                deltaigr = shape[3]
                deltacov = shape[4]
                print("************ Processing Shape ",s,"***************")
                filename = "Objects/Benchmark/{}.xyz".format(s)
    
                pc, nc = from_xyz_normals(filename)
                M=1
    
                vertices, edges, triangles, net, skpts, upts = build_neural_skeleton(pc, nc, tv = True, activation="Sine", npl=64, dep=6, hints=10000, delta=delta, time_limit=time_limit)
                torch.save(net, "Networks/net_{}_{}_{}_{}.net".format(npl, dep, "Sine"+("" if tv else "NoTV"), s))
                to_obj(vertices*M, "Results_benchmark/cvsk_{}_{}.obj".format("Sine"+("" if tv else "NoTV"), s), lines=edges, tri=triangles)
                to_obj(upts*M, "Results_benchmark/unif_points_{}_{}.obj".format("Sine", s))
                to_obj(skpts*M, "Results_benchmark/skeletal_points_{}_{}.obj".format("Sine", s))
    
                vertices_siren, edges_siren, triangles_siren, net_siren, skpts_siren, upts_siren = build_siren_neural_skeleton(pc, nc, npl=64, dep=6, delta=deltasiren, time_limit=time_limit)
                torch.save(net_siren, "Networks/net_{}_{}_{}_{}.net".format(npl, dep, "siren", s))
                to_obj(vertices_siren*M, "Results_benchmark/cvsk_{}_{}.obj".format("siren", s), lines=edges_siren, tri=triangles_siren)
                to_obj(upts_siren*M, "Results_benchmark/unif_points_{}_{}.obj".format("siren", s))
                to_obj(skpts_siren*M, "Results_benchmark/skeletal_points_{}_{}.obj".format("siren", s))
    
                vertices_igr, edges_igr, triangles_igr, net_igr, skpts_igr, upts_igr = build_igr_neural_skeleton(pc, nc, npl=64, dep = 6, delta=deltaigr, time_limit=time_limit)
                torch.save(net_igr, "Networks/net_{}_{}_{}_{}.net".format(npl, dep, "igr", s))
                to_obj(vertices_igr*M, "Results_benchmark/cvsk_{}_{}.obj".format("igr",s), lines=edges_igr, tri=triangles_igr)
                to_obj(upts_igr*M, "Results_benchmark/unif_points_{}_{}.obj".format("igr", s))
                to_obj(skpts_igr*M, "Results_benchmark/skeletal_points_{}_{}.obj".format("igr", s))
        
                vertices_cov, edges_cov, triangles_cov, covcandidates = build_coverage_skeleton(pc, nc, delta = deltacov, npts = 10000, time_limit=time_limit)
                to_obj(covcandidates*M, "Results_benchmark/cov_cand_{}.obj".format(s))
                to_obj(vertices_cov*M, "Results_benchmark/cvsk_{}_{}.obj".format("coverage", s), lines=edges_cov, tri=triangles_cov)
                
    print("skeletons computed in ", '{:.2f}'.format(time.time()-tskel),"s.")
    
    print("Computing distances to groundtruth")
    tdist = time.time()
    val=[]
    i = 0
    for shape in shapes:
        s = shape[0]
        print("****************{}****************".format(s))
        print("Ours")
        cdn, hdn = compare_skeletal_points_to_gtskel(gt, "Results_benchmark/skeletal_points_{}_{}.obj".format("Sine",s), mindist = mindist)
        print("IGR")
        cdigr, hdigr = compare_skeletal_points_to_gtskel(gt, "Results_benchmark/skeletal_points_{}_{}.obj".format("igr",s), mindist = mindist)
        print("Coverage")
        cdc, hdc = compare_skeletal_points_to_gtskel(gt, "Results_benchmark/cov_cand_{}.obj".format(s), mindist=mindist)
        print("Siren")
        cdsiren, hdsiren = compare_skeletal_points_to_gtskel(gt, "Results_benchmark/skeletal_points_{}_{}.obj".format("siren", s), mindist=mindist)
        print("MCS")
        cdmcs, hdmcs = 0,0#compare_skeletal_points_to_gtskel(gt, "MCS/ResultsPoints/skeletal_points_MCS_{}.skel.obj".format(s), mindist=mindist, verbose=False)
        print("Voxel Cores")
        cdvx, hdvx = 0,0#compare_skeletal_points_to_gtskel(gt, "comparison_results/voxelcores/skeletal_points_{}_surf_voxelcore_thinned0_02.obj".format(s), mindist = mindist, verbose=False)

        tabledata.append([ s, "{0:.2g}/{1:.2g}".format(cdn,hdn), "{0:.2g}/{1:.2g}".format(cdsiren,hdsiren), "{0:.2g}/{1:.2g}".format(cdigr,hdigr), "{0:.2g}/{1:.2g}".format(cdc,hdc), "{0:.2g}/{1:.2g}".format(cdmcs,hdmcs), "{0:.2g}/{1:.2g}".format(cdvx,hdvx)])

        #tcdc[i]+=0.1*cdc
        #thdc[i]+=0.1*hdc
        #tcdn[i]+=0.1*cdn
        #thdn[i]+=0.1*hdn
        #tcdrelu[i]+=0.1*cdrelu
        #thdrelu[i]+=0.1*hdrelu
        #tcdsiren[i]+=0.1*cdsiren
        #thdsiren[i]+=0.1*hdsiren
    
        i = i+1
    res.append([val])
    print("distances computed in ", '{:.2f}'.format(time.time()-tdist),"s.")


#for i in range(0,8):
#    tabledata.append([ s, "{0:.4g}/{1:.4g}".format(cdn[i],hdn[i]),
#                         "{0:.4g}/{1:.4g}".format(cdsiren[i],hdsiren[i]),
#                         "{0:.4g}/{1:.4g}".format(cdrelu[i],hdrelu[i]),
#                         "{0:.4g}/{1:.4g}".format(cdc[i],hdc[i]),
#                         "{0:.4g}/{1:.4g}".format(cdmdc[i],hdmdc[i])])

col_names=["Shape", "Ours", "SIREN", "IGR", "Cov. Axis", "MCS", "Voxel Cores"]

print(tabulate(tabledata, headers=col_names, tablefmt="fancy_grid"))
with open('table_benchmark.tex', 'w') as f: f.write(tabulate(tabledata, headers=col_names, tablefmt="latex"))
