import torch
import numpy as np

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
    # pts: 2BxNx3 or 2BxNx2 for SIFT
    # desc: 2Bx256xN or 2BX128XN for SIFT
    # output: 
    # corr: BxNx4 correspondence
    # matching_score: BxN matching score

    # compute distance matrix
    batch_size, channel, N = desc.shape
    pts = pts.view(batch_size//2, 2, N, -1)
    pts1, pts2 = pts[:,0,:,:], pts[:,1,:,:]
    desc = desc.view(batch_size//2, 2, channel, N)
    desc1, desc2 = desc[:,0,:,:], desc[:,1,:,:]
    distmat = pair_dist(desc1, desc2)
    matching_score, argmin = distmat.min(dim=-1)
    batch_idx = torch.arange(batch_size//2).view(-1,1).repeat(1,N).view(-1)
    pts2_match = pts2[batch_idx, argmin.view(-1), :].view(batch_size//2, N, -1)
    corr = torch.cat([pts1[:,:,:2], pts2_match[:,:,:2]],dim=-1)
    return corr, matching_score



