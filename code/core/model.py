import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('../network')
sys.path.append('../util')
from superpoint import SuperPointNet, KeypointExtract
from oanet import OANet, weighted_8points
from nn_match import NearestNeighborMatch
from geometry import norm_kpt


class Model(torch.nn.Module):
  def __init__(self, config):
    nn.Module.__init__(self)
    self.sp = SuperPointNet()
    self.extract = KeypointExtract(config)
    self.oan = OANet(config)
    if config.pretrain_backbone is not None:
        print('load pretrain superpoint from '+config.pretrain_backbone)
        self.sp.load_state_dict(torch.load(config.pretrain_backbone))
    if config.pretrain_corrnet is not None:
        print('load pretrain corrnet from '+config.pretrain_corrnet)
        model_best = torch.load(config.pretrain_corrnet)
        self.oan.load_state_dict(model_best['state_dict'])
    if config.fix_backbone:
        # fix superpoint
        for param in self.sp.parameters():
            param.requires_grad = False
    if config.fix_temp:
        for param in self.extract.parameters():
            param.requires_grad = False


  def forward(self, data):
    # output: 
    # det_logits: 2B*65*H*W
    # kpt_desc: 2B*C*N
    # corr: B*1*N*4
    # corr_logits: B*N
    # e_hat: B*9

    det_logits, coarse_desc = self.sp(data['imgs'])
    pts, kpt_desc, heatmap = self.extract(det_logits, coarse_desc)
    if pts is not None:
        corr, match_score = NearestNeighborMatch(pts, kpt_desc)
        # corr: B*N*4
        corr[:,:,:2] = norm_kpt(corr[:,:,:2], data['K1s'])
        corr[:,:,2:4] = norm_kpt(corr[:,:,2:4], data['K2s'])
        corr = corr.unsqueeze(1)
        corr_logits = self.oan(corr)
        e_hat = weighted_8points(corr, corr_logits)
        return det_logits, kpt_desc, corr, corr_logits, e_hat
    else:
        return [None]*5


class CorrModel(torch.nn.Module):
  def __init__(self, config):
    nn.Module.__init__(self)
    self.oan = OANet(config)

  def forward(self, data):
    logits = self.oan(data['kpt'])
    e_hat = weighted_8points(data['kpt'], logits, data['T1s'], data['T2s'])
    return data['kpt'], None, logits, e_hat


