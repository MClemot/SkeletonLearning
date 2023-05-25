import time
import torch
from skeleton import coverage_skeleton, voronoi_candidates
from geometry import center_bounding_box

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_coverage_skeleton(pc1, nc1, delta = 0.06, npts = 10000,time_limit=120):
    pc, nc = torch.tensor(pc1, device=device).float(), torch.tensor(nc1, device=device).float()

    pc,center,scale = center_bounding_box(pc)
    pc.requires_grad = True
    t = time.time()

    idx = torch.randint(high=pc.shape[0], size=(npts,))
    pc = pc[idx]
    nc = nc[idx]

    candidates = voronoi_candidates(pc, nc, reduce_radius=0.01)

    cvskpts, edges, triangles = coverage_skeleton(candidates, pc, delta=delta, time_limit=time_limit)

    if cvskpts.shape[0] == 0 :
        print("Infeasible problem no skeleton found, try tweaking the parameters")
        print("Coverage skeleton failed after ", time.time()-t,"s.")
    else:
        print("Coverage skeleton built in ", '{:.2f}'.format(time.time()-t),"s.")
        cvskpts = cvskpts * scale.cpu().numpy() + center.cpu().numpy()

    candidates = candidates.cpu().numpy() * scale.cpu().numpy() + center.cpu().numpy()

    return cvskpts, edges, triangles, candidates