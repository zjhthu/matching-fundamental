import torch
import numpy as np
import argparse
import os
import glob
from tqdm import tqdm
from scipy import sparse
import pickle
import cv2
import h5py
import sys
sys.path.append('../network')
sys.path.append('../data')
from superpoint import SuperPointNet, depth2space, get_kpt, spatial_soft_argmax
from img_data import read_image


def str2bool(v):
    return v.lower() in ("true", "1")
# Parse command line arguments.
parser = argparse.ArgumentParser(description='PyTorch SuperPoint Demo.')
parser.add_argument('input', type=str, default='',
  help='Image directory or movie file or "camera" (for webcam).')
parser.add_argument('--img_glob', type=str, default='*.png',
  help='Glob match if directory of images is specified (default: \'*.png\').')
parser.add_argument('--H', type=int, default=480,
  help='Input image height (default: 480).')
parser.add_argument('--W', type=int, default=640,
  help='Input image width (default:640).')
parser.add_argument('--weights_path', type=str, default='../../model/superpoint_v1.pth',
  help='pretrained superpoint weights')
parser.add_argument('--reshape', type=str2bool, default=False,
  help='whether reshape when read image')
parser.add_argument('--save_only_detector', type=str2bool, default=False,
  help='only save heatmap')
parser.add_argument('--nms_dist', type=int, default=3,
  help='Non Maximum Suppression (NMS) distance (default: 3).')
parser.add_argument('--conf_thresh', type=float, default=0.015,
  help='Detector confidence threshold (default: 0.015).')
parser.add_argument('--save_prefix', type=str, default='sp-th015-nms3',
  help='prefix of filename.')
parser.add_argument('--num_kpt', type=int, default=2048,
  help='save kpt number')
parser.add_argument('--bord', type=int, default=4,
  help='boader remove')
parser.add_argument('--use_spatial_softmax', type=str2bool, default=False,
  help='use spatial softmax to refine kpt')
parser.add_argument('--spatial_window', type=int, default=7,
  help='window size in spatial softmax')
parser.add_argument('--spatial_temp', type=float, default=20,
  help='temperature in spatial softmax')
parser.add_argument('--use_sift_kpt', type=str2bool, default=False,
  help='use sift kpt rather than superpoint')
parser.add_argument('--save_hpatch', type=str2bool, default=False,
  help='save for hpatch evaluation')
parser.add_argument('--save_corr', type=str2bool, default=False,
  help='save for corr learning')

opt = parser.parse_args()

if opt.use_sift_kpt:
  sift = cv2.xfeatures2d.SIFT_create(nfeatures=opt.num_kpt, contrastThreshold=1e-5)
# get image lists
search = os.path.join(opt.input, opt.img_glob)
listing = glob.glob(search)

net = SuperPointNet()
print('load pretrain superpoint from '+opt.weights_path)
net.load_state_dict(torch.load(opt.weights_path))
net.cuda()
net.eval()

img_size = [opt.W, opt.H]

def get_kpt_feature(coarse_desc, batch_pts):
	# --- Process descriptor.
    D = coarse_desc.shape[1] # B*C*Hc*Wc
    _, H, W = heatmap.shape

    # Interpolate into descriptor map using 2D point locations.
    samp_pts = batch_pts[:,:,[0,1]]
    samp_pts[:,:,0] = (samp_pts[:,:,0] / (float(W)/2.)) - 1.
    samp_pts[:,:,1] = (samp_pts[:,:,1] / (float(H)/2.)) - 1.
    samp_pts = samp_pts.view(1, 1, -1, 2)
    desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts).squeeze(2) # B*C*min_pts
    desc = torch.nn.functional.normalize(desc, dim=1)
    return desc

for img_name in tqdm(listing):
    with torch.no_grad():
        img = read_image(img_name, reshape=opt.reshape, img_size=img_size)
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).cuda()
        semi, coarse_desc = net(img)
        heatmap = depth2space(semi)# 1*H*W
        if opt.save_only_detector:
            label = (heatmap.squeeze(0) > opt.conf_thresh).detach().cpu().numpy()
            sparse.save_npz(img_name[:-4]+'-'+opt.save_prefix, sparse.csr_matrix(label))
        else:
            if not opt.use_sift_kpt:
              # pts : 3xN
              pts = get_kpt((heatmap[0].detach().cpu().numpy(), opt.conf_thresh, opt.nms_dist, opt.bord, opt.num_kpt))
            else:
              img_rgb = cv2.imread(img_name)
              cv_kp = sift.detect(img_rgb, None)
              pts = np.array([_kp.pt for _kp in cv_kp]).T

            if opt.use_spatial_softmax:
                offset = np.linspace(-(opt.spatial_window//2),opt.spatial_window//2,num=(opt.spatial_window//2)*2+1)
                offset = np.meshgrid(offset,offset)
                offset = torch.from_numpy(np.concatenate((offset[0].reshape(-1,1), offset[1].reshape(-1,1)), axis=1)).long()

                batch_pts = torch.from_numpy(np.concatenate((np.zeros([1,pts.shape[1]]), pts), axis=0)).cuda().reshape(4,1,-1).float()
                batch_pts = spatial_soft_argmax(batch_pts, heatmap, opt.spatial_window, offset, opt.spatial_temp)# batch*min_pts*2
            else:
                batch_pts = torch.from_numpy(pts[:2,:].T.reshape([1,-1,2])).cuda().float()

            desc = get_kpt_feature(coarse_desc, batch_pts)
            if opt.save_hpatch:
              # to numpy
              pts = batch_pts.squeeze(0).detach().cpu().numpy() # Nx2
              desc = desc.squeeze(0).detach().cpu().numpy().T # Nx256
              desc_file = img_name[:-4]+'-'+opt.save_prefix+'.pkl'
              pickle.dump({'npy_kpts':pts, 'mul_s_patches':None, 'sift_desc':desc}, open(desc_file,'wb'), protocol=2)
            if opt.save_corr:
              desc_file = img_name + ".numkp2000."+opt.save_prefix+".desc.h5"
              pts = batch_pts.squeeze(0).detach().cpu().numpy().T # 2xNx
              desc = desc.squeeze(0).detach().cpu().numpy() # 256xN
              with h5py.File(desc_file, "w") as ifp:
                  ifp.create_dataset('keypoints', pts.shape, dtype=np.float32)
                  ifp.create_dataset('descriptors', desc.shape, dtype=np.float32)
                  ifp["keypoints"][:] = pts
                  ifp["descriptors"][:] = desc








