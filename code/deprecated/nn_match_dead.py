import torch
import numpy as np
import scipy.spatial

def torch_squareform(a):
        # currently 2019.04.12 pytorch do not support squareform, 
        # so we rely on the scipy.spatial.distance.squareform
        # luckily we do not need autograd here
        # input: 
        # a: batch*(N-1)N/2
        # output:
        # b: batch*N*N
        b = []
        for batch_idx in range(a.shape[0]):
                b.append(scipy.spatial.distance.squareform (a[batch_idx].detach().numpy())) 
        return torch.from_numpy(np.stack(b)).to(a.device)


def test_squareform():
        import time
        a = torch.randn(32,500,256)
        time1 = time.time()
        distmat1 = torch.norm(a.unsqueeze(1) - a.unsqueeze(2), dim=3)
        print('implmentation1 time '+str(time.time()-time1))
        time2 = time.time()
        dist = torch.pdist(a)
        distmat2 = torch_squareform(dist)
        print('implmentation2 time '+str(time.time()-time2))
        time3 = time.time()
        b = (a**2).sum(2)
        c = (a**2).sum(2)
        d = torch.matmul(a, a.transpose(1,2))
        #distmat3 = torch.pow(b.unsqueeze(1)+c.unsqueeze(2)-2*d, 0.5)
        distmat3 = torch.sqrt(b.unsqueeze(1)+c.unsqueeze(2)-2*d)
        distmat3[torch.isnan(distmat3)] = 0
        print('implmentation3 time '+str(time.time()-time3))
        assert (distmat1-distmat2).abs().max() < 1e-5
        #import pdb;pdb.set_trace()
        #import IPython;IPython.embed() 
        print((distmat1-distmat3).abs().max())
        assert (distmat1-distmat3).abs().max() < 5e-2

def pair_dist(a,b):
    # input:
    # a: batch*channel*N1
    # b: batch*channel*N2
    # output: batch*N1*N2
    a2 = (a**2).sum(1)
    b2 = (b**2).sum(1)
    return a2.unsqueeze(2)+b2.unsqueeze(1)-2*torch.matmul(a.transpose(1,2),b)

def test_pdist():
    import time
    a = torch.rand(32,256,500)
    b = torch.rand(32,256,600)
    time1 = time.time()
    distmat1 = torch.norm(a.unsqueeze(3)-b.unsqueeze(2),dim=1)
    print('implmentation1 time '+str(time.time()-time1))
    time2 = time.time()
    distmat2 = pair_dist(a,b)
    print('implmentation2 time '+str(time.time()-time2))
    print((distmat1**2 - distmat2).abs().max())
    assert (distmat1**2 - distmat2).abs().max() < 1e-3

#test_pdist()

        
def NearestNeighborMatch(pts, desc):
        # input:
        # pts: BxNx3 or BxNx2 for SIFT
        # desc: Bx256xN or BX128XN for SIFT
        # output: 
        # corr: BxNx4 correspondence
        # score: BxN matching score

        # compute distance matrix
        dist = torch.pdist(desc.transpose(1,2).contiguous())
        distmat = torch_squareform(dist)

