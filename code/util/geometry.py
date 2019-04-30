import torch
import cv2
import numpy as np
import sys
sys.path.append('../util')
from transformations import quaternion_from_matrix

# --------------------- torch functions -------------------------
def norm_kpt(kpt, K):
    # input:
    # kpt: B*N*2
    # K: B*3*3
    # output:
    # kpt_norm: B*N*2
    batch_size, N, _ = kpt.shape
    kpt = torch.cat([kpt.transpose(1,2), torch.ones(batch_size, 1, N).to(kpt.device)], dim=1)
    kpt_norm = torch.matmul(torch.inverse(K), kpt)
    return kpt_norm[:,:2,:].transpose(1,2)

def skew_symmetric(v):

    zero = torch.zeros_like(v[:, 0])

    M = torch.stack([
        zero, -v[:, 2], v[:, 1],
        v[:, 2], zero, -v[:, 0],
        -v[:, 1], v[:, 0], zero,
    ], dim=1)

    return M
# --------------------- torch functions end ----------------------


# --------------------- numpy functions --------------------------
def unnorm_kpt(xy, K):
    K = K.reshape(3,3)
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    return xy * np.asarray([fx, fy]) + np.array([cx, cy])

def compute_error_for_find_essential(x1, x2, E):
    # x1.shape == x2.shape == (Np, 3)
    Ex1 = E.dot(x1.T)
    Etx2 = E.T.dot(x2.T)

    Ex1 = E.dot(x1.T)
    Etx2 = E.T.dot(x2.T)
    x2tEx1 = np.sum(x2.T * Ex1, axis=0)

    a = Ex1[0] * Ex1[0]
    b = Ex1[1] * Ex1[1]
    c = Etx2[0] * Etx2[0]
    d = Etx2[1] * Etx2[1]

    err = x2tEx1 * x2tEx1 / (a + b + c + d)

    return err


def ourFindEssentialMat(np1, np2, method=cv2.RANSAC, iter_num=1000,
                        threshold=0.01, probs=None, weighted=False,
                        use_prob=True):
    """Python implementation of OpenCV findEssentialMat.

    We have this to try multiple different options for RANSAC, e.g. MLESAC,
    which we simply resorted to doing nothing at the end and not using this
    function.

    """

    min_pt_num = 5
    Np = np1.shape[0]
    perms = np.arange(Np, dtype=np.int)

    best_E = None
    best_inliers = None
    best_err = np.inf

    _np1 = np.concatenate([np1, np.ones((Np, 1))], axis=1)
    _np2 = np.concatenate([np2, np.ones((Np, 1))], axis=1)

    thresh2 = threshold * threshold

    for n in range(iter_num):
        # Randomly select depending on the probability (if given)
        if probs is not None:
            probs /= np.sum(probs)
        if use_prob:
            cur_subs = np.random.choice(
                perms, min_pt_num, replace=False, p=probs)
        else:
            cur_subs = np.random.choice(
                perms, min_pt_num, replace=False, p=None)

        sub_np1 = np1[cur_subs, :]
        sub_np2 = np2[cur_subs, :]
        Es, mask = cv2.findEssentialMat(
            sub_np1, sub_np2, focal=1, pp=(0, 0), method=cv2.RANSAC)
        if Es is None:
            # print('E is None @ {} iteration'.format(n))
            continue

        for i in range(0, Es.shape[0], 3):
            E = Es[i:i + 3, :]
            err = compute_error_for_find_essential(_np1, _np2, E)

            inliers = err <= thresh2
            if method == cv2.RANSAC:
                if weighted:
                    num_inliers = (inliers * probs).sum()
                else:
                    num_inliers = inliers.sum()
                sum_err = -num_inliers
            elif method == 'MLESAC':  # worse than RANSAC
                if weighted:
                    sum_err = (np.abs(err) * probs).sum()
                else:
                    sum_err = np.abs(err).sum()
            if sum_err < best_err:
                best_err = sum_err
                best_E = E
                best_inliers = inliers

    best_inliers = best_inliers.reshape(-1, 1).astype(np.uint8)

    return best_E, best_inliers


def evaluate_R_t(R_gt, t_gt, R, t, q_gt=None):

    # from Utils.transformations import quaternion_from_matrix

    t = t.flatten()
    t_gt = t_gt.flatten()

    eps = 1e-15

    if q_gt is None:
        q_gt = quaternion_from_matrix(R_gt)
    q = quaternion_from_matrix(R)
    q = q / (np.linalg.norm(q) + eps)
    q_gt = q_gt / (np.linalg.norm(q_gt) + eps)
    loss_q = np.maximum(eps, (1.0 - np.sum(q * q_gt)**2))
    err_q = np.arccos(1 - 2 * loss_q)

    # dR = np.dot(R, R_gt.T)
    # dt = t - np.dot(dR, t_gt)
    # dR = np.dot(R, R_gt.T)
    # dt = t - t_gt
    t = t / (np.linalg.norm(t) + eps)
    t_gt = t_gt / (np.linalg.norm(t_gt) + eps)
    loss_t = np.maximum(eps, (1.0 - np.sum(t * t_gt)**2))
    err_t = np.arccos(np.sqrt(1 - loss_t))

    if np.sum(np.isnan(err_q)) or np.sum(np.isnan(err_t)):
        # This should never happen! Debug here
        import IPython
        IPython.embed()

    return err_q, err_t


def eval_nondecompose(p1s, p2s, E_hat, dR, dt, scores):

    # Use only the top 10% in terms of score to decompose, we can probably
    # implement a better way of doing this, but this should be just fine.
    num_top = len(scores) // 10
    num_top = max(1, num_top)
    th = np.sort(scores)[::-1][num_top]
    mask = scores >= th

    p1s_good = p1s[mask]
    p2s_good = p2s[mask]

    # Match types
    E_hat = E_hat.reshape(3, 3).astype(p1s.dtype)
    R, t = None, None
    if p1s_good.shape[0] >= 5:
        # Get the best E just in case we get multipl E from findEssentialMat
        num_inlier, R, t, mask_new = cv2.recoverPose(
            E_hat, p1s_good, p2s_good)
        try:
            err_q, err_t = evaluate_R_t(dR, dt, R, t)
        except:
            print("Failed in evaluation")
            print(R)
            print(t)
            err_q = np.pi
            err_t = np.pi / 2
            import pdb;pdb.set_trace()
    else:
        err_q = np.pi
        err_t = np.pi / 2

    loss_q = np.sqrt(0.5 * (1 - np.cos(err_q)))
    loss_t = np.sqrt(1.0 - np.cos(err_t)**2)

    # Change mask type
    mask = mask.flatten().astype(bool)

    mask_updated = mask.copy()
    if mask_new is not None:
        # Change mask type
        mask_new = mask_new.flatten().astype(bool)
        mask_updated[mask] = mask_new

    return err_q, err_t, loss_q, loss_t, np.sum(num_inlier), mask_updated, R, t


def eval_decompose(p1s, p2s, dR, dt, mask=None, method=cv2.LMEDS, probs=None,
                   weighted=False, use_prob=True):
    if mask is None:
        mask = np.ones((len(p1s),), dtype=bool)
    # Change mask type
    mask = mask.flatten().astype(bool)

    # Mask the ones that will not be used
    p1s_good = p1s[mask]
    p2s_good = p2s[mask]
    probs_good = None
    if probs is not None:
        probs_good = probs[mask]

    num_inlier = 0
    mask_new2 = None
    R, t = None, None
    if p1s_good.shape[0] >= 5:
        if probs is None and method != "MLESAC":
            E, mask_new = cv2.findEssentialMat(
                p1s_good, p2s_good, method=method, threshold=0.01)
        else:
            E, mask_new = ourFindEssentialMat(
                p1s_good, p2s_good, method=method, threshold=0.01,
                probs=probs_good, weighted=weighted, use_prob=use_prob)
        if E is not None:
            new_RT = False
            # Get the best E just in case we get multipl E from
            # findEssentialMat
            for _E in np.split(E, len(E) / 3):
                _num_inlier, _R, _t, _mask_new2 = cv2.recoverPose(
                    _E, p1s_good, p2s_good, mask=mask_new)
                if _num_inlier > num_inlier:
                    num_inlier = _num_inlier
                    R = _R
                    t = _t
                    mask_new2 = _mask_new2
                    new_RT = True
            if new_RT:
                err_q, err_t = evaluate_R_t(dR, dt, R, t)
            else:
                err_q = np.pi
                err_t = np.pi / 2

        else:
            err_q = np.pi
            err_t = np.pi / 2
    else:
        err_q = np.pi
        err_t = np.pi / 2

    loss_q = np.sqrt(0.5 * (1 - np.cos(err_q)))
    loss_t = np.sqrt(1.0 - np.cos(err_t)**2)

    mask_updated = mask.copy()
    if mask_new2 is not None:
        # Change mask type
        mask_new2 = mask_new2.flatten().astype(bool)
        mask_updated[mask] = mask_new2

    return err_q, err_t, loss_q, loss_t, np.sum(num_inlier), mask_updated, R, t


# --------------------- numpy functions end ----------------------
