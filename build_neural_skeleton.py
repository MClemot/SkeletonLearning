import numpy as np
import time
import torch

from nn import optimize_neural_sdf
from skeleton import sample_skeleton_gpu, coverage_skeleton, neural_candidates
from geometry import projection, uniform_resampling, center_bounding_box, get_bounding_box

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_neural_skeleton(pc1, nc1,
                          tv = True, activation="Sine", npl=64, dep=6,
                          hints=1000, delta=0.06, scaling=None,
                          lambda_pc = 100, lambda_eik=2e2, lambda_hint=1e2, lambda_tv=2e1,
                          resampling=True, trainednet=None, time_limit=120, scaleshape=True):

    print(delta)
    tinit = time.time()
    
    if trainednet != None:
        net = torch.load(trainednet)
        pc, nc = torch.tensor(pc1, device=device).float(), torch.tensor(nc1, device=device).float()
        t = time.time()
        if scaleshape == True:
            center,scale = get_bounding_box(pc)
        else:
            center = torch.zeros((3), device=device)
            scale = torch.ones((1),device=device)
    else:
        pc, nc = torch.tensor(pc1, device=device).float(), torch.tensor(nc1, device=device).float()
    
        if scaleshape == True:
            pc,center,scale = center_bounding_box(pc)
        else:
            center = torch.zeros((3), device=device)
            scale = torch.ones((1),device=device)
        pc.requires_grad = True
        
        net = torch.load("Pretrained/pretrained_{}_{}_{}.net".format(npl, dep, activation))
        
        print("\n##### Optimizing the neural sdf ({},{})".format(activation, "TV" if tv else "No TV"))
        if activation=="Sine" :
            optim = torch.optim.LBFGS(params=net.parameters())
            nepochs=50
            nhints_ends=20
        elif activation=="ReLU":
            optim = torch.optim.Adam(params=net.parameters(), lr=2e-5)
            nepochs=20000
            nhints_ends=10000
        elif activation=="SoftPlus":
            optim = torch.optim.Adam(params=net.parameters(), lr=1e-3)
            nepochs=20000
            nhints_ends=10000
        
        t = time.time()
      
        try:
            optimize_neural_sdf(net, optim, pc, nc,
                                batch_size=25000, pc_batch_size=25000, 
                                epochs=nepochs, tv_ends=nepochs if tv else 0, hints_ends=nhints_ends,# 20 20 5
                                lambda_pc = lambda_pc, lambda_eik=lambda_eik, lambda_hint=lambda_hint, lambda_tv=lambda_tv,
                                nb_hints=hints, plot_loss=False)
        except KeyboardInterrupt:
            pass
        print("Optimizing NN took", '{:.2f}'.format(time.time()-t),"s.")
    
    print("\n##### Computing neural coverage skeleton")
    tskel = time.time()


    
    D = 2*np.sqrt(3)
    number = 10000
    samples = projection(net, number=number, prune=True)
    if resampling:
        samples = uniform_resampling(net, samples, 100, K=3, alpha=.1*np.sqrt(D/number), sigma=16*D/number)


    sk = sample_skeleton_gpu(net, samples, res=50, length=1, steps=1, div=100)
    
    print("Extracting skeletal points candidates", time.time()-t)
    
    candidates = neural_candidates(sk, reduce_radius=0.01)
    
    cvskpts, edges, triangles = coverage_skeleton(candidates, samples, delta=delta, factor=scaling, time_limit=time_limit)

    print("Coverage skeleton obtained in", '{:.2f}'.format(time.time()-tskel)," s.") 

    #putting the samples back to their original position and scale
    skpts = candidates.cpu().numpy() * scale.cpu().numpy() + center.cpu().numpy()
    upts = samples.detach().cpu().numpy() * scale.cpu().numpy() + center.cpu().numpy()

    print("Total computation time", '{:.2f}'.format(time.time()-tinit)," s.")



    if cvskpts.shape[0] == 0 :
        print("Infeasible problem no skeleton found, try tweaking the parameters")
        print("Coverage skeleton failed after", time.time()-t,"s.")

    else:
        print("Coverage skeleton built in", '{:.2f}'.format(time.time()-t),"s.")
        #putting the skeletal points back to their original position and scale
        cvskpts = cvskpts * scale.cpu().numpy() + center.cpu().numpy()

    return cvskpts,edges,triangles,net,skpts,upts
