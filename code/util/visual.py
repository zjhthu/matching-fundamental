import cv2
import numpy as np
from geometry import unnorm_kpt


def visual_corr(img, x1, x2, mask_gt, mask, prefix):
    mask = np.where(mask)[0].tolist()
    mask_gt = mask_gt[mask]
    # inlier
    inlier_idx = np.where(mask_gt)[0]
    inlier = [mask[idx] for idx in inlier_idx]
    color = (0,255,0)
    for pt_idx in inlier:
        cv2.circle(img, tuple(x1[pt_idx,:].astype(int)), 1, color)
        cv2.circle(img, tuple(x2[pt_idx,:].astype(int)), 1, color)
        cv2.line(img, tuple(x1[pt_idx, :].astype(int)), tuple(x2[pt_idx, :].astype(int)), color, 3)
    # outlier
    outlier_idx = np.where(~mask_gt)[0]
    outlier = [mask[idx] for idx in outlier_idx]
    color = (0,0,255)
    for pt_idx in outlier:
        cv2.circle(img, tuple(x1[pt_idx,:].astype(int)), 1, color)
        cv2.circle(img, tuple(x2[pt_idx,:].astype(int)), 1, color)
        cv2.line(img, tuple(x1[pt_idx, :].astype(int)), tuple(x2[pt_idx, :].astype(int)), color, 1)
    cv2.imwrite(prefix+'.png', img)
def visual_pair(mask_gt, mask, x, img1, img2, K1, K2, prefix):
    # img: C*H*W
    assert img1.ndim == 3 and (img1.shape[0] == 3 or img1.shape[0] == 1)
    if img1.shape[0] == 1:
        # from superpoint
        img1 = np.tile(img1, (3,1,1))*255
        img2 = np.tile(img2, (3,1,1))*255
    offset = img1.shape[1]
    x1 = unnorm_kpt(x[:,:2], K1)
    x2 = unnorm_kpt(x[:,2:], K2) + np.asarray([0, offset])
    img_concat = np.ascontiguousarray(np.concatenate([img1, img2],axis=1).transpose(1,2,0))
    visual_corr(img_concat, x1, x2, mask_gt, mask, prefix)
