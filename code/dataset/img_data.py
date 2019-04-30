import cv2
import numpy as np
import sys
import torch
import torch.utils.data
import scipy.sparse 

sys.path.append('../util')
from io_util import loadpkl

def collate_fn(batch):
    batch_size = len(batch)
    data = {}
    data['imgs'], data['kpt_map'], data['K1s'], data['K2s'], data['Rs'], data['ts'] = [], [], [], [], [], []
    for sample in batch:
        data['imgs'].append(sample['img1'])
        data['imgs'].append(sample['img2'])
        data['kpt_map'].append(sample['kpt_map1'])
        data['kpt_map'].append(sample['kpt_map2'])
        data['K1s'].append(sample['K1'])
        data['K2s'].append(sample['K2'])
        data['Rs'].append(sample['R'])
        data['ts'].append(sample['t'])

    for key in ['imgs', 'K1s', 'K2s', 'Rs', 'ts']:
        data[key] = torch.from_numpy(np.stack(data[key])).float()
    data['kpt_map'] = torch.from_numpy(np.stack(data['kpt_map'])).long()
    data['imgs'] = data['imgs'].unsqueeze(1)
    # for key in data.keys():
    #     data[key] = data[key].cuda()
    return data

def read_image(impath, reshape = False, img_size=None):
    """ Read image as grayscale and resize to img_size.
        Inputs
            impath: Path to input image.
            img_size: (W, H) tuple specifying resize size.
        Returns
            grayim: float32 numpy array sized H x W with values in range [0, 1].
    """
    grayim = cv2.imread(impath, 0)
    if grayim is None:
        raise Exception('Error reading image %s' % impath)
    if reshape:
        # Image is resized via opencv.
        interp = cv2.INTER_AREA
        grayim = cv2.resize(grayim, (img_size[0], img_size[1]), interpolation=interp)
    else:
        H, W = grayim.shape[0], grayim.shape[1]
        cell = 8
        Hc = int(H / cell) * cell
        Wc = int(W / cell) * cell
        grayim[:Hc, :Wc]
    grayim = (grayim.astype('float32') / 255.)
    return grayim

class MatchingDataset(torch.utils.data.Dataset):
    def __init__(self, filename, config):
        self.data = loadpkl(filename)
        self.config = config
        # print(len(self.data))

    def __getitem__(self, index):
        # (i,j) (name_i, name_j), (geo_i, geo_j)
        data_item = self.data[index]
        img1_name = self.config.dataset_path+'/'+data_item[1][0]
        img2_name = self.config.dataset_path+'/'+data_item[1][1]
        #print(img1_name)
        #print(img2_name)
        K1, K2 = data_item[2][0]['K'], data_item[2][1]['K']
        # original image size: w, h
        img1_size, img2_size = data_item[2][0]['imsize'].reshape(2).astype(np.float32), data_item[2][1]['imsize'].reshape(2).astype(np.float32)
        # add cx, cy
        K1[0,2] += (img1_size[0] - 1.0) * 0.5
        K1[1,2] += (img1_size[1] - 1.0) * 0.5
        K2[0,2] += (img2_size[0] - 1.0) * 0.5
        K2[1,2] += (img2_size[1] - 1.0) * 0.5
        # compute relative R and t
        R1, R2 = data_item[2][0]['R'].reshape([3, 3]), data_item[2][1]['R'].reshape([3, 3])
        t1, t2 = data_item[2][0]['T'].reshape([3, 1]), data_item[2][1]['T'].reshape([3, 1])
        R = np.dot(R2, R1.T)
        t = t2 - np.dot(R, t1)

        # reshape size 
        img_size = [self.config.img_W, self.config.img_H]
        img1 = read_image(img1_name, reshape=True, img_size=img_size)
        img2 = read_image(img2_name, reshape=True, img_size=img_size)
        # scale
        S1 = np.diag([float(img_size[0])/img1_size[0], float(img_size[1])/img1_size[1], 1])
        S2 = np.diag([float(img_size[0])/img2_size[0], float(img_size[1])/img2_size[1], 1])
        K1, K2 = np.dot(S1, K1), np.dot(S2, K2)

        # keypoint map
        kpt_map1 = scipy.sparse.load_npz(img1_name[:-4]+'-'+self.config.kpt_prefix+'.npz').todense()
        kpt_map2 = scipy.sparse.load_npz(img2_name[:-4]+'-'+self.config.kpt_prefix+'.npz').todense()
        # space to depth
        Hc, Wc = int(self.config.img_H/self.config.cell), int(self.config.img_W/self.config.cell)
        kpt_map1 = np.asarray(kpt_map1).reshape([Hc, self.config.cell, Wc, self.config.cell]).transpose([0,2,1,3]).reshape([Hc,Wc,-1])
        kpt_map1 = np.argmax(np.concatenate([kpt_map1*2, np.ones([Hc,Wc,1])], axis=-1), axis=-1)
        kpt_map2 = np.asarray(kpt_map2).reshape([Hc, self.config.cell, Wc, self.config.cell]).transpose([0,2,1,3]).reshape([Hc,Wc,-1])
        kpt_map2 = np.argmax(np.concatenate([kpt_map2*2, np.ones([Hc,Wc,1])], axis=-1), axis=-1)

        return {'img1':img1, 'img2':img2, 'K1':K1, 'K2':K2, 'R':R, 't':t, 'kpt_map1':kpt_map1, 'kpt_map2':kpt_map2}

        # else:
        #     img1, img2 = cv2.imread(img1_name), cv2.imread(img2_name)
        #     cv_kp1, desc1 = self.sift.detectAndCompute(img1, None) # N*128
        #     cv_kp2, desc2 = self.sift.detectAndCompute(img2, None)
        #     pts1 = np.array([_kp.pt for _kp in cv_kp1])# N*2
        #     pts2 = np.array([_kp.pt for _kp in cv_kp2])
        #     import pdb;pdb.set_trace()
        #     return {'pts1':pts1,'pts2':pts2,'desc1':desc1,'desc2':desc2,'K1':K1, 'K2':K2, 'R':R, 't':t}

    def __len__(self):
        return len(self.data)






