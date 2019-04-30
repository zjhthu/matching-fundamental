
from __future__ import print_function
import sys
sys.path.append('../')
import itertools
from six.moves import xrange
import os
import torch
import numpy as np
import cv2
from config import get_config, print_usage
config, unparsed = get_config()
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
from network_insert import Model, weighted_8points, compute_loss
from network_pt import Model as ModelPointNet
from geometry import eval_nondecompose, evaluate_R_t
from geom import get_episym, parse_geom
import math
import matplotlib.pyplot as plt
from data import loadFromDir



# copy from tests.py
# return inlier after recover pose
def eval_sample(_xs, _ys, _dR, _dt, e_hat_out, y_hat_out, config, test_list):

    if len(y_hat_out) != _xs.shape[1]:
        print('set y hat to ones')
        exit(0)
    # Eval decompose for all pairs
    _xs = _xs.reshape(-1, 4)
    # x coordinates
    _x1 = _xs[:, :2]
    _x2 = _xs[:, 2:]
    # current validity from network
    _valid = y_hat_out.flatten()
    # choose top ones (get validity threshold)
    _valid_th = np.sort(_valid)[::-1][config.obj_top_k]
    _relu_tanh = np.maximum(0, np.tanh(_valid))

    # For every things to test
    _use_prob = True
    for _test in test_list:
        if _test == "ours":
            _eval_func = "non-decompose"
            _mask_before = _valid >= max(0, _valid_th)
            _method = None
            _probs = None
            _weighted = False
        elif _test == "ours_ransac":
            _eval_func = "decompose"
            _mask_before = _valid >= max(0, _valid_th)
            _method = cv2.RANSAC
            _probs = None
            _weighted = False

        if _eval_func == "non-decompose":
            _err_q, _err_t, _, _, _num_inlier, \
                mask_updated = eval_nondecompose(
                    _x1, _x2, e_hat_out, _dR, _dt, y_hat_out)
            _mask_after = mask_updated# modified here, different from test.py
        elif _eval_func == "decompose":
            # print("RANSAC loop with ours")
            _err_q, _err_t, _, _, _num_inlier, \
                _mask_after, e_hat_out = eval_decompose(
                    _x1, _x2, _dR, _dt, mask=_mask_before,
                    method=_method, probs=_probs,
                    weighted=_weighted, use_prob=_use_prob)

    return _mask_before, _mask_after,  _err_q, _err_t

def unnormlize(x, cx, cy, fx, fy, offset):
    return x * np.asarray([fx, fy]) + np.array([cx, cy+offset])
def visual_corr(img, x1, x2, mask_gt, mask,  prefix):
    #import pdb;pdb.set_trace()
    mask = np.where(mask)[0].tolist()
    mask_gt = mask_gt[mask]
    # inlier
    inlier_idx = np.where(mask_gt)[0]
    inlier = [mask[idx] for idx in inlier_idx]
    color = (0,255,0)
    for pt_idx in inlier:
        cv2.circle(img, tuple(x1[pt_idx,:].astype(int)), 1, color)
        cv2.circle(img, tuple(x2[pt_idx,:].astype(int)), 1, color)
        cv2.line(img, tuple(x1[pt_idx, :].astype(int)), tuple(x2[pt_idx, :].astype(int)), color, 1)
    # outlier
    outlier_idx = np.where(~mask_gt)[0]
    outlier = [mask[idx] for idx in outlier_idx]
    color = (0,0,255)
    for pt_idx in outlier:
        cv2.circle(img, tuple(x1[pt_idx,:].astype(int)), 1, color)
        cv2.circle(img, tuple(x2[pt_idx,:].astype(int)), 1, color)
        cv2.line(img, tuple(x1[pt_idx, :].astype(int)), tuple(x2[pt_idx, :].astype(int)), color, 1)
    cv2.imwrite(prefix+'.png', img)
    print('inlier '+str(len(inlier))+' outlier '+str(len(outlier)))

def visual_pair(mask_gt, mask, x, img1, img2, cx1, cy1, cx2, cy2, f1, f2, prefix):
    offset = img1.shape[1]
    x1 = unnormlize(x[:,:2], cx1, cy1, f1, f1, 0)
    x2 = unnormlize(x[:,2:], cx2, cy2, f2, f2, offset)
    img_concat = np.ascontiguousarray(np.concatenate([img1, img2],axis=1).transpose(1,2,0))
    visual_corr(img_concat, x1, x2, mask_gt, mask, prefix)
 
def visual_pts(x, img1, img2, cx1, cy1, cx2, cy2, f1, f2, prefix):
    offset = img1.shape[1]
    x1 = unnormlize(x[:,:2], cx1, cy1, f1, f1, 0)
    x2 = unnormlize(x[:,2:], cx2, cy2, f2, f2, offset)
    img_concat = np.ascontiguousarray(np.concatenate([img1, img2],axis=1).transpose(1,2,0))
    # visual_corr(img_concat, x1, x2, mask_gt, mask, prefix)
    color = (0,0,255)
    for pt_idx in range(x.shape[0]):
        cv2.circle(img_concat, tuple(x1[pt_idx,:].astype(int)), 1, color)
        cv2.circle(img_concat, tuple(x2[pt_idx,:].astype(int)), 1, color)
        cv2.line(img_concat, tuple(x1[pt_idx, :].astype(int)), tuple(x2[pt_idx, :].astype(int)), color, 1)
    cv2.imwrite(prefix+'.png', img_concat)


def load_img(config, data_dir): 
    img, geom, vis, depth, kp, desc = loadFromDir(
        data_dir,
        "-16x16",
        bUseColorImage=True,
        crop_center=config.data_crop_center,
        load_lift=config.use_lift)
    return img, geom, vis

sift = cv2.xfeatures2d.SIFT_create(
nfeatures=config.obj_num_kp, contrastThreshold=1e-5)
# compute input and label
def make_xy(idx_i, idx_j, imgs, geom, geom_type):
    # randomly shuffle the pairs and select num_sample amount
    np.random.seed(1234)
    def extractkp(idx, img, geom, geom_type):
        cv_kp, cv_desc = sift.detectAndCompute(img.transpose(
            1, 2, 0), None)
        img_out = img.copy()
        img_out=cv2.drawKeypoints(img.transpose(1, 2, 0),cv_kp,img_out)
        # cv2.imwrite('sift_keypoints.png',img_out)

        cx = (img[0].shape[1] - 1.0) * 0.5
        cy = (img[0].shape[0] - 1.0) * 0.5
        # Correct coordinates using K
        cx = parse_geom(geom, geom_type)["K"][idx, 0, 2]
        cy = parse_geom(geom, geom_type)["K"][idx, 1, 2]
        xy = np.array([_kp.pt for _kp in cv_kp])
        # Correct focals
        fx = parse_geom(geom, geom_type)["K"][idx, 0, 0]
        fy = parse_geom(geom, geom_type)["K"][idx, 1, 1]
        kp = (
            xy - np.array([[cx, cy]])
        ) / np.asarray([[fx, fy]])
        desc = cv_desc
        if np.isclose(fx, fy):
            f = fx
        else:
            f = (fx, fy)
        return kp, desc, cx, cy, f       
    def compute_idx_sort(desc_ii, desc_jj):
        # compute decriptor distance matrix
        distmat = np.sqrt(
            np.sum(
                (np.expand_dims(desc_ii, 1) - np.expand_dims(desc_jj, 0))**2,
                axis=2))
        # Choose K best from N
        idx_sort = np.argsort(distmat, axis=1)[:, :config.obj_num_nn]
        idx_sort = (
            np.repeat(
                np.arange(distmat.shape[0])[..., None],
                idx_sort.shape[1], axis=1
            ),
            idx_sort
        )
        return idx_sort
    def getxy(idx_i, idx_j, kpi, kpj, idx_sort, geom, geom_type):
        # ------------------------------
        # Get dR
        R_i = parse_geom(geom, geom_type)["R"][idx_i]
        R_j = parse_geom(geom, geom_type)["R"][idx_j]
        dR = np.dot(R_j, R_i.T)
        # Get dt
        t_i = parse_geom(geom, geom_type)["t"][idx_i].reshape([3, 1])
        t_j = parse_geom(geom, geom_type)["t"][idx_j].reshape([3, 1])
        dt = t_j - np.dot(dR, t_i)
        # ------------------------------
        # Get sift points for the first image
        x1 = kpi
        x2 = kpj 
        # ------------------------------
        # create x1, y1, x2, y2 as a matrix combo
        x1mat = np.repeat(x1[:, 0][..., None], len(x2), axis=-1)
        y1mat = np.repeat(x1[:, 1][..., None], len(x2), axis=1)
        x2mat = np.repeat(x2[:, 0][None], len(x1), axis=0)
        y2mat = np.repeat(x2[:, 1][None], len(x1), axis=0)
        # Move back to tuples
        idx_sort = (idx_sort[0], idx_sort[1])
        x1mat = x1mat[idx_sort]
        y1mat = y1mat[idx_sort]
        x2mat = x2mat[idx_sort]
        y2mat = y2mat[idx_sort]
        # Turn into x1, x2
        x1 = np.concatenate(
            [x1mat.reshape(-1, 1), y1mat.reshape(-1, 1)], axis=1)
        x2 = np.concatenate(
            [x2mat.reshape(-1, 1), y2mat.reshape(-1, 1)], axis=1)
        # make xs in NHWC
        xs = np.concatenate([x1, x2], axis=1).T.reshape(4, 1, -1).transpose((1, 2, 0))
        # ------------------------------
        # Get the geodesic distance using with x1, x2, dR, dt
        geod_d = get_episym(x1, x2, dR, dt)
        ys = geod_d
        # Save R, t for evaluation
        Rs = np.array(dR).reshape(3, 3)
        # normalize t before saving
        dtnorm = np.sqrt(np.sum(dt**2))
        assert (dtnorm > 1e-5)
        dt /= dtnorm
        ts = np.array(dt).flatten() 
        return xs, ys, Rs, ts

    imgi = imgs[idx_i]
    imgj = imgs[idx_j]
    kpi, desci, cxi, cyi, fi = extractkp(idx_i, imgi, geom, geom_type)
    kpj, descj, cxj, cyj, fj = extractkp(idx_j, imgj, geom, geom_type)
    idx_sort = compute_idx_sort(desci, descj)
    xs, ys, Rs, ts = getxy(idx_i, idx_j, kpi, kpj, idx_sort, geom, geom_type)
    return xs, ys, Rs, ts, imgi, imgj, cxi, cyi, cxj, cyj, fi, fj


imgs, geom, vis = load_img(config, '/home/jhzhang/datasets/te-mit4/test/')

pairs = []
for ii, jj in itertools.product(xrange(len(imgs)), xrange(len(imgs))):
    if ii != jj and imgs[ii].shape[2] == imgs[jj].shape[2] and vis[ii][jj] > 0.35:
        pairs.append((ii, jj))
print('total '+str(len(pairs))+' pairs')
pairs = pairs[np.random.permutation(len(pairs))[:1000]]

geom_type = 'Calibration'


# load model
model = Model(config)
save_file_best = os.path.join(config.model_path, 'model_best.pth')
if not os.path.exists(save_file_best):
    print("Model File {} does not exist! Quiting".format(save_file_best))
    exit(1)
# Restore model
checkpoint = torch.load(save_file_best)
start_epoch = checkpoint['epoch']
model.load_state_dict(checkpoint['state_dict'])
model.cuda()
print("Restoring from " + str(save_file_best) + ', ' + str(start_epoch) + "epoch...\n")
model.eval()


inlier_our=[]
inlier_pointnet = []
inlier_ransac=[]
label_our=[]
label_pointnet = []
label_ransac=[]
topks = []

# eval for each sample
test_list = ['ours']
with torch.no_grad():
    for idx in range(pairs.shape[0]):
    #for idx in [1]:
        print('idx '+str(idx))
        _xs, _ys, _dR, _dt, img1, img2, cx1, cy1, cx2, cy2, f1, f2 = make_xy(pairs[idx][0], pairs[idx][1], imgs, geom, geom_type)
        _xs_cpu, _ys_cpu, _dR_cpu, _dt_cpu = _xs, _ys, _dR, _dt
        _xs, _ys, _dR, _dt = torch.Tensor(_xs.reshape(1,1,-1,4)).cuda(), torch.Tensor(_ys.reshape(1,-1,1)).cuda(), \
                torch.Tensor(_dR.reshape(1,9)).cuda(), torch.Tensor(_dt.reshape(1,3)).cuda()
        # inlier gt
        import pdb;pdb.set_trace()
        mask_gt = (_ys[:,:,0] < config.obj_geod_th*100).cpu().numpy().flatten().astype(bool)
        
        # eval our model
        y_hat,topk = model(_xs)
        topks.append(topk)
        e_hat = weighted_8points(_xs, y_hat)
        y_hat, e_hat = y_hat.cpu().numpy().flatten(), e_hat.cpu().numpy().flatten()
        our_before, our_after, our_after2, err_q, err_t = eval_sample(_xs_cpu, _ys_cpu, _dR_cpu, _dt_cpu, e_hat, y_hat, config, ['ours'])
        label_before, label_after, label_after2 = mask_gt[np.where(our_before)[0]], mask_gt[np.where(our_after)[0]], mask_gt[np.where(our_after2)[0]]
        
        inlier_our.append(float(label_after.sum())/(label_after.size+1e-10))
        label_our.append(our_after)
        visual_pair(mask_gt, our_after, _xs_cpu.squeeze(0), img1, img2, cx1, cy1, cx2, cy2, f1, f2, 'our-'+str(idx)+'-'+str(pairs[idx][0])+'-'+str(pairs[idx][1]))

                
        # eval pointnet
        y_hat_pointnet = model_pointnet(_xs)
        e_hat_pointnet = weighted_8points(_xs, y_hat_pointnet)
        y_hat_pointnet, e_hat_pointnet = y_hat_pointnet.cpu().numpy().flatten(), e_hat_pointnet.cpu().numpy().flatten()
        pointnet_before, pointnet_after, pointnet_after2, err_q, err_t = eval_sample(_xs_cpu, _ys_cpu, _dR_cpu, _dt_cpu, e_hat_pointnet, y_hat_pointnet, config, ['ours'])
        label_before, label_after, label_after2 = mask_gt[np.where(pointnet_before)[0]], mask_gt[np.where(pointnet_after)[0]], mask_gt[np.where(pointnet_after2)[0]]
        
        inlier_pointnet.append(float(label_after.sum())/(label_after.size+1e-10))
        label_pointnet.append(pointnet_after)
        visual_pair(mask_gt, pointnet_after, _xs_cpu.squeeze(0), img1, img2, cx1, cy1, cx2, cy2, f1, f2, 'pointnet-'+str(idx)+'-'+str(pairs[idx][0])+'-'+str(pairs[idx][1]))


        #eval ransac
        ransac_before, ransac_after, ransac_after2, err_q, err_t = eval_sample(_xs_cpu, _ys_cpu, _dR_cpu, _dt_cpu, None, np.ones(_ys_cpu.shape), config, ['ours_ransac'])
        label_before, label_after, label_after2 = mask_gt[np.where(ransac_before)[0]], mask_gt[np.where(ransac_after)[0]], mask_gt[np.where(ransac_after2)[0]]
        inlier_ransac.append(float(label_after2.sum())/(label_after2.size+1e-10))
        label_ransac.append(ransac_after)
        visual_pair(mask_gt, ransac_after, _xs_cpu.squeeze(0), img1, img2, cx1, cy1, cx2, cy2, f1, f2, 'ransac-'+str(idx)+'-'+str(pairs[idx][0])+'-'+str(pairs[idx][1]))
        

