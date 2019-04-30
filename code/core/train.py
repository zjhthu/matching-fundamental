import numpy as np
import torch
import torch.optim as optim
from tqdm import trange
import os
import time
from logger import Logger
from test import valid
from loss import geo_loss, detector_loss
import sys
sys.path.append('../util')
from util import tocuda
from visual import visual_pair

def adjust_learning_rate(optimizer, epoch, config):
    lr = config.train_lr * (config.gamma ** (epoch // config.lr_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train_step(step, optimizer, model, data, config):
    model.train()
    det_logits, kpt_desc, corr, corr_logits, e_hat = model(data)
    if corr is not None:
        import pdb;pdb.set_trace()
        loss_geo, gt_geod_d, loss_ess_val, loss_corr_cla_val = geo_loss(step, corr, data['Rs'], data['ts'], corr_logits, e_hat, config)
        #loss_det, loss_det_val = detector_loss(det_logits, data['kpt_map'], config)

        '''
        for batch_idx in range(data['K1s'].shape[0]):
            mask_gt = (gt_geod_d[batch_idx] < config.vis_geod_th).detach().cpu().numpy()
            #print('inlier '+str(mask_gt.sum()))
            mask = np.ones(mask_gt.shape) > 0
            visual_pair(mask_gt, mask, corr[batch_idx,0,:,:].detach().cpu().numpy(), \
                data['imgs'][batch_idx*2,:,:,:].detach().cpu().numpy(), \
                data['imgs'][batch_idx*2+1,:,:,:].detach().cpu().numpy(), \
                data['K1s'][batch_idx,:,:].detach().cpu().numpy(), \
                data['K2s'][batch_idx,:,:].detach().cpu().numpy(), \
                str(batch_idx)
                )
        #import pdb;pdb.set_trace()
        '''
        #loss = loss_geo+loss_det
        loss = loss_geo
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    else:
        print('no keypoint, skip')
        return [-1]*3
    return loss_ess_val, loss_corr_cla_val, loss_det_val


def train(model, train_loader, valid_loader, config):
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=config.train_lr, weight_decay = config.weight_decay)

    checkpoint_path = os.path.join(config.log_path, 'checkpoint.pth')
    resume = os.path.isfile(checkpoint_path)
    if resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(checkpoint_path)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger_train = Logger(os.path.join(config.log_path, 'log_train.txt'), resume=True)
        logger_valid = Logger(os.path.join(config.log_path, 'log_valid.txt'), resume=True)
    else:
        best_acc = -1
        start_epoch = 0
        logger_train = Logger(os.path.join(config.log_path, 'log_train.txt'))
        logger_train.set_names(['Learning Rate', 'Essential Loss', 'Classfi Loss', 'Detector loss'])
        logger_valid = Logger(os.path.join(config.log_path, 'log_valid.txt'))
        logger_valid.set_names(['Valid Acc', 'Essential Loss', 'Clasfi Loss'])
    train_loader_iter = iter(train_loader)
    for step in trange(start_epoch, config.train_iter, ncols=config.tqdm_width):
        try:
            train_data = next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(train_loader)
            train_data = next(train_loader_iter)

        train_data = tocuda(train_data)
        # run training
        cur_lr = adjust_learning_rate(optimizer, step, config) 
        loss_val = train_step(step, optimizer, model, train_data, config)
        logger_train.append([cur_lr]+list(loss_val))

        # Check if we want to write validation
        b_save = ((step + 1) % config.save_intv) == 0
        b_validate = ((step + 1) % config.val_intv) == 0
        if b_validate:
            va_res, loss1, loss2 = valid(valid_loader, step, config)
            logger_valid.append([va_res, loss1, loss2])
            if va_res > best_acc:
                print("Saving best model with va_res = {}".format(va_res))
                best_acc = va_res
                torch.save({
                'epoch': step + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
                }, os.path.join(config.log_path, 'model_best.pth'))

        if b_save:
            torch.save({
            'epoch': step + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
            }, checkpoint_path)


