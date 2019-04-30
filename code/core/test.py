import torch
import numpy as np
import os
from six.moves import xrange
import cv2
import sys
sys.path.append('../util')
sys.path.append('../network')
from geometry import eval_nondecompose, eval_decompose
from loss import geo_loss
from visual import visual_pair
from util import tocuda, get_pool_result


def test_sample(args):
    _xs, _dR, _dt, _e_hat, _y_hat, cur_i, img1, img2, _y_gt, K1, K2, config = args
    _xs = _xs.reshape(-1, 4).astype('float64')
    _dR, _dt = _dR.astype('float64').reshape(3,3), _dt.astype('float64')
    _y_hat_out = _y_hat.flatten().astype('float64')
    e_hat_out = _e_hat.flatten().astype('float64')

    _x1 = _xs[:, :2]
    _x2 = _xs[:, 2:]
    # current validity from network
    _valid = _y_hat_out
    # choose top ones (get validity threshold)
    _valid_th = np.sort(_valid)[::-1][config.obj_top_k]
    _mask_before = _valid >= max(0, _valid_th)

    if not config.use_ransac:
        _err_q, _err_t, _, _, _num_inlier, \
        _mask_updated, _R_hat, _t_hat = eval_nondecompose(
                _x1, _x2, e_hat_out, _dR, _dt, _y_hat_out)

    else:
        # actually not use prob here since probs is None
        _use_prob = True
        _method = cv2.RANSAC
        _probs = None
        _weighted = False
        _err_q, _err_t, _, _, _num_inlier, \
        _mask_updated, _R_hat, _t_hat = eval_decompose(
                    _x1, _x2, _dR, _dt, mask=_mask_before,
                    method=_method, probs=_probs,
                    weighted=_weighted, use_prob=_use_prob)
    if config.save_vis:
        mask_gt = (_y_gt < config.vis_geod_th).flatten()
        save_prefix = config.res_path + '/' + str(cur_i)
        visual_pair(mask_gt, _mask_updated.flatten(), _xs, img1, img2, K1, K2, save_prefix)
        visual_pair(mask_gt, _mask_before.flatten(), _xs, img1, img2, K1, K2, save_prefix+'-before')
    '''
    print('err_q')
    print(_err_q*180/np.pi)
    print('err_t')
    print(_err_t*180/np.pi)
    import pdb;pdb.set_trace()
    '''
    return [float(_err_q), float(_err_t), float(_num_inlier), _R_hat.reshape(1,-1), _t_hat.reshape(1,-1)]

def test_process(mode, model, cur_global_step, data_loader, config):
    model.eval()
    loader_iter = iter(data_loader)
    if not config.use_ransac:
        _tag = "ours"
    else:
        _tag = "ours_ransac"

    if config.res_path == '':
        config.res_path = os.path.join(config.log_path[:-5], mode)

    l1s, l2s, results, pool_arg = [], [], [], []
    eval_step, eval_step_i, num_processor = 100, 0, 8
    t_gt = []
    with torch.no_grad(): 
        for test_data in loader_iter:
            test_data = tocuda(test_data)
            det_logits, kpt_desc, corr, y_hat, e_hat = model(test_data)
            loss, gt_geod_d, l1, l2 = geo_loss(cur_global_step, corr, test_data['Rs'], test_data['ts'], y_hat, e_hat, config)
            l1s.append(l1)
            l2s.append(l2)
            # get essential matrix from fundamental matrix
            e_hat = torch.matmul(torch.matmul(test_data['K2s'].transpose(1,2), e_hat.reshape(-1,3,3)),test_data['K1s']).reshape(-1,9)

            t_gt.append(test_data['ts'].reshape(1,-1).detach().cpu().numpy())
            for batch_idx in range(e_hat.shape[0]):
                img1, img2 = None, None
                if config.save_vis:
                    img1, img2 = test_data['imgs'][2*batch_idx], test_data['imgs'][2*batch_idx+1]
                pool_arg += [(corr[batch_idx].detach().cpu().numpy(), test_data['Rs'][batch_idx].detach().cpu().numpy(), \
                              test_data['ts'][batch_idx].detach().cpu().numpy(), e_hat[batch_idx].detach().cpu().numpy(), \
                              y_hat[batch_idx].detach().cpu().numpy(),  \
                              eval_step_i, img1, img2, gt_geod_d[batch_idx].detach().cpu().numpy(), \
                              test_data['K1s'].detach().cpu().numpy(), test_data['K2s'].detach().cpu().numpy(), \
                              config)]
                #test_sample(pool_arg[0])
                #import pdb;pdb.set_trace()
                eval_step_i += 1
                if eval_step_i % eval_step == 0:
                    results += get_pool_result(num_processor, test_sample, pool_arg)
                    pool_arg = []
        if len(pool_arg) > 0:
            results += get_pool_result(num_processor, test_sample, pool_arg)

    measure_list = ["err_q", "err_t", "num", 'R_hat', 't_hat']
    eval_res = {}
    for measure in measure_list:
        eval_res[measure] =  np.zeros(len(results))
    eval_res["err_q"] = np.asarray([result[0] for result in results])
    eval_res["err_t"] = np.asarray([result[1] for result in results])
    eval_res["num"] = np.asarray([result[2] for result in results])
    eval_res["R_hat"] = np.asarray([result[3] for result in results])
    eval_res["t_hat"] = np.asarray([result[4] for result in results])
    t_gt = np.asarray(t_gt)


    # dump test results
    ret_val = 0
    for _sub_tag in measure_list:
        # For median error
        ofn = os.path.join(
            config.res_path, "median_{}_{}.txt".format(_sub_tag, _tag))
        with open(ofn, "w") as ofp:
            ofp.write("{}\n".format(np.median(eval_res[_sub_tag])))

    ths = np.arange(7) * 5
    cur_err_q = np.array(eval_res["err_q"]) * 180.0 / np.pi
    cur_err_t = np.array(eval_res["err_t"]) * 180.0 / np.pi
    # Get histogram
    q_acc_hist, _ = np.histogram(cur_err_q, ths)
    t_acc_hist, _ = np.histogram(cur_err_t, ths)
    qt_acc_hist, _ = np.histogram(np.maximum(cur_err_q, cur_err_t), ths)
    num_pair = float(len(cur_err_q))
    q_acc_hist = q_acc_hist.astype(float) / num_pair
    t_acc_hist = t_acc_hist.astype(float) / num_pair
    qt_acc_hist = qt_acc_hist.astype(float) / num_pair
    q_acc = np.cumsum(q_acc_hist)
    t_acc = np.cumsum(t_acc_hist)
    qt_acc = np.cumsum(qt_acc_hist)
    # Store return val
    for _idx_th in xrange(1, len(ths)):
        ofn = os.path.join(config.res_path, "acc_q_auc{}_{}.txt".format(ths[_idx_th], _tag))
        with open(ofn, "w") as ofp:
            ofp.write("{}\n".format(np.mean(q_acc[:_idx_th])))
        ofn = os.path.join(config.res_path, "acc_t_auc{}_{}.txt".format(ths[_idx_th], _tag))
        with open(ofn, "w") as ofp:
            ofp.write("{}\n".format(np.mean(t_acc[:_idx_th])))
        ofn = os.path.join(config.res_path, "acc_qt_auc{}_{}.txt".format(ths[_idx_th], _tag))
        with open(ofn, "w") as ofp:
            ofp.write("{}\n".format(np.mean(qt_acc[:_idx_th])))


    ofn = os.path.join(config.res_path, "all_acc_qt_auc20_{}.txt".format(_tag))
    np.savetxt(ofn, np.maximum(cur_err_q, cur_err_t))
    ofn = os.path.join(config.res_path, "all_acc_q_auc20_{}.txt".format(_tag))
    np.savetxt(ofn, cur_err_q)
    ofn = os.path.join(config.res_path, "all_acc_t_auc20_{}.txt".format(_tag))
    np.savetxt(ofn, cur_err_t)
    ofn = os.path.join(config.res_path, "all_t_gt.txt")
    np.savetxt(ofn, t_gt.squeeze(1))
    ofn = os.path.join(config.res_path, "all_t_hat.txt")
    np.savetxt(ofn, eval_res["t_hat"].squeeze(1))


    # Return qt_auc20 
    ret_val = np.mean(qt_acc[:4])  # 1 == 5
    return [ret_val, np.mean(np.asarray(l1s)), np.mean(np.asarray(l2s))]


def test(data_loader, model, config):
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
    va_res = test_process("test", model, 0, data_loader, config)
    print('test result '+str(va_res))
def valid(data_loader, model, step, config):
    config.use_ransac = False
    return test_process("valid", model, step, data_loader, config)

