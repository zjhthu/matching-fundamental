import torch
import numpy as np
from geometry import skew_symmetric

# TODO: need to be tested
def batch_get_episym(x1,x2,E):
    # input: 
    # x1,x2: B*N*2
    # E: B*9
    # output:
    # ys: B*N
    assert x1.dim() == 3
    #import pdb;pdb.set_trace()
    batch_size, num_pts = x1.shape[0], x1.shape[1]
    x1 = torch.cat([x1, x1.new_ones(batch_size, num_pts,1)], dim=-1).reshape(batch_size, num_pts,3,1)
    x2 = torch.cat([x2, x2.new_ones(batch_size, num_pts,1)], dim=-1).reshape(batch_size, num_pts,3,1)
    F = E.reshape(-1,1,3,3).repeat(1,num_pts,1,1)
    x2Fx1 = torch.matmul(x2.transpose(2,3), torch.matmul(F, x1)).reshape(batch_size,num_pts)
    Fx1 = torch.matmul(F,x1).reshape(batch_size,num_pts,3)
    Ftx2 = torch.matmul(F.transpose(2,3),x2).reshape(batch_size,num_pts,3)
    ys = x2Fx1**2 * (
            1.0 / (Fx1[:, :, 0]**2 + Fx1[:, :, 1]**2) +
            1.0 / (Ftx2[:, :, 0]**2 + Ftx2[:, :, 1]**2))
    return ys

def compute_gt_geod(corr, e_gt):
    # input:
    # corr: B*1*N*4
    # e_gt: B*9
    # output: B*N
    # The groundtruth epi sqr
    gt_geod_d = batch_get_episym(corr[:,0,:,:2], corr[:,0,:,2:4], e_gt)
    return gt_geod_d


def geo_loss(global_step, corr, R_in, t_in, K1s, K2s, logits, e_hat, config):

    # Get groundtruth Essential matrix
    e_gt_unnorm = torch.matmul(
        torch.reshape(skew_symmetric(t_in), (-1, 3, 3)),
        torch.reshape(R_in, (-1, 3, 3))
        )
    # get groundtruth fundamental matrix
    e_gt_unnorm = torch.matmul(torch.matmul(torch.inverse(K2s).transpose(1,2), e_gt_unnorm), K1s).reshape(-1,9)
    e_gt = e_gt_unnorm / torch.norm(e_gt_unnorm, dim=1, keepdim=True)

    # Compute groundtruth for classification
    gt_geod_d = compute_gt_geod(corr, e_gt)
    #import pdb;pdb.set_trace()

    # Essential matrix loss
    essential_loss = torch.mean(torch.min(
        torch.sum(torch.pow(e_hat - e_gt, 2), dim=1),
        torch.sum(torch.pow(e_hat + e_gt, 2), dim=1)
    ))
    '''
    import pdb;pdb.set_trace()
    print('e_hat')
    print(e_hat.reshape(3,3))
    print('e_gt')
    print(e_gt.reshape(3,3))
    print('-')
    print((e_hat-e_gt).reshape(3,3))
    print(torch.norm(e_hat-e_gt)**2)
    print('+')
    print((e_hat+e_gt).reshape(3,3))
    print(torch.norm(e_hat+e_gt)**2)
    '''

    # Classification loss
    is_pos = (gt_geod_d < config.obj_geod_th).type(logits.type())
    is_neg = (gt_geod_d >= config.obj_geod_th).type(logits.type())
    #neg_idx = torch.nonzero(is_neg)
    #for idx in neg_idx:
    #    corr[idx[0],0,idx[1],:] = corr[idx[0],0,idx[1],:].detach()
        
    #print('pos '+str(is_pos.sum())+' neg '+str(is_neg.sum()))
    c = is_pos - is_neg
    classif_losses = -torch.log(torch.sigmoid(c * logits) + np.finfo(float).eps.item())

    # balance
    num_pos = torch.relu(torch.sum(is_pos, dim=1) - 1.0) + 1.0
    num_neg = torch.relu(torch.sum(is_neg, dim=1) - 1.0) + 1.0
    classif_loss_p = torch.sum(classif_losses * is_pos, dim=1)
    classif_loss_n = torch.sum(classif_losses * is_neg, dim=1)
    classif_loss = torch.mean(classif_loss_p * 0.5 / num_pos + classif_loss_n * 0.5 / num_neg)

    # precision = torch.mean(
    #     torch.sum((logits > 0).type(is_pos.type()) * is_pos, dim=1) /
    #     torch.sum((logits > 0).type(is_pos.type()) * (is_pos + is_neg), dim=1)
    # )
    # recall = torch.mean(
    #     torch.sum((logits > 0).type(is_pos.type()) * is_pos, dim=1) /
    #     torch.sum(is_pos, dim=1)
    # )
    loss = 0
    # Check global_step and add essential loss
    if config.loss_essential > 0:
        loss += (config.loss_essential * essential_loss * float(global_step >= config.loss_essential_init_iter))
    if config.loss_classif > 0:
        loss += config.loss_classif * classif_loss
    #print(essential_loss)

    return [loss, gt_geod_d, (config.loss_essential * essential_loss).item(), (config.loss_classif * classif_loss).item()]

def detector_loss(logits, label, config):
    # input:
    # logits: 2Bx65xHcxWc
    # label: 2BxHcxWc
    if not config.fix_backbone:
        loss =  torch.nn.functional.cross_entropy(logits, label)*config.loss_detector
        return [loss, loss.item()]
    else:
        return [0, 0]




