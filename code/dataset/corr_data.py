import numpy as np
import sys
import torch
import glob
import os
import pickle
import h5py

def collate_fn(batch):
    batch_size = len(batch)
    numkps = np.array([sample['xs'].shape[1] for sample in batch])
    cur_num_kp = int(numkps.min())

    data = {}
    data['imgs'], data['K1s'], data['K2s'], data['Rs'], \
        data['ts'], data['kpt'], data['ys'], data['T1s'], data['T2s']  = [], [], [], [], [], [], [], [], []
    for sample in batch:
        data['imgs'].append(sample['img1'])
        data['imgs'].append(sample['img2'])
        data['K1s'].append(sample['K1'])
        data['K2s'].append(sample['K2'])
        data['T1s'].append(sample['T1'])
        data['T2s'].append(sample['T2'])
        data['Rs'].append(sample['R'])
        data['ts'].append(sample['t'])
        if sample['xs'].shape[1] > cur_num_kp:
            sub_idx = np.random.choice(sample['xs'].shape[1], cur_num_kp)
            data['kpt'].append(sample['xs'][:,sub_idx,:])
            data['ys'].append(sample['ys'][sub_idx,:])
        else:
            data['kpt'].append(sample['xs'])
            data['ys'].append(sample['ys'])

    for key in ['K1s', 'K2s', 'Rs', 'ts', 'kpt', 'ys', 'T1s', 'T2s']:
        data[key] = torch.from_numpy(np.stack(data[key])).float()
    return data

class MatchingDataset(torch.utils.data.Dataset):
    def __init__(self, filename, config):
        # self.data = self.load_data(filename)
        self.config = config
        self.filename = filename
        self.data = None

    def __getitem__(self, index):
        if self.data is None:
            self.data = h5py.File(self.filename,'r')

        #index = 169
        img1 = None
        img2 = None
        cx1 = np.asarray(self.data['cx1s'][str(index)])
        cy1 = np.asarray(self.data['cy1s'][str(index)])
        cx2 = np.asarray(self.data['cx2s'][str(index)])
        cy2 = np.asarray(self.data['cy2s'][str(index)])
        f1 = np.asarray(self.data['f1s'][str(index)])
        f2 = np.asarray(self.data['f2s'][str(index)])
        xs = np.asarray(self.data['xs'][str(index)])
        ys = np.asarray(self.data['ys'][str(index)])
        # if type(f1) != tuple:
        #     f1 = (f1,f1)
        # if type(f2) != tuple:
        #     f2 = (f2,f2)  
        R = np.asarray(self.data['Rs'][str(index)])
        t = np.asarray(self.data['ts'][str(index)])
        K1 = np.asarray([
            [f1[0], 0, cx1[0]],
            [0, f1[1], cy1[0]],
            [0, 0, 1]
            ])
        K2 = np.asarray([
            [f2[0], 0, cx2[0]],
            [0, f2[1], cy2[0]],
            [0, 0, 1]
            ])
        if self.config.input_noise > 0:
            N = xs.shape[1]
            noise = np.random.normal(0, self.config.input_noise, N*4).reshape(N,4)
            normalized_noise = (noise / np.asarray([f1[0], f1[1], f2[0], f2[1]])).reshape(1,N,4)
            xs = xs + normalized_noise
        # normalize data
        x1, x2 = xs[0,:,:2], xs[0,:,2:4]
        x1 = x1 * np.asarray([K1[0,0], K1[1,1]]) + np.array([K1[0,2], K1[1,2]])
        x2 = x2 * np.asarray([K2[0,0], K2[1,1]]) + np.array([K2[0,2], K2[1,2]])
        x1_mean, x2_mean = np.mean(x1, axis=1), np.mean(x2, axis=1)
        x1_std, x2_std = np.std(x1, axis=(0,1)), np.std(x2, axis=(0,1))
        T1, T2 = np.zeros([3,3]), np.zeros([3,3])
        T1[0,0], T1[1,1], T1[2,2] = np.sqrt(2)/x1_std, np.sqrt(2)/x1_std, 1
        T1[0,2], T1[1,2] = -np.sqrt(2)/x1_std*x1_mean[0], -np.sqrt(2)/x1_std*x1_mean[1]
        T2[0,0], T2[1,1], T2[2,2] = np.sqrt(2)/x2_std, np.sqrt(2)/x2_std, 1
        T2[0,2], T2[1,2] = -np.sqrt(2)/x2_std*x2_mean[0], -np.sqrt(2)/x2_std*x2_mean[1]
        x1 = np.dot(np.concatenate([x1, np.ones([x1.shape[0],1])], axis=-1), T1.T)[:,:2]
        x2 = np.dot(np.concatenate([x2, np.ones([x2.shape[0],1])], axis=-1), T2.T)[:,:2]
        xs = np.concatenate([x1,x2],axis=-1).reshape(1,-1,4)

        return {'img1':img1, 'img2':img2, 'K1':K1, 'K2':K2, 'R':R, 't':t, 'xs':xs, 'ys':ys, 'T1':T1, 'T2':T2}

    def __len__(self):
        if self.data is None:
            self.data = h5py.File(self.filename,'r')
            _len = len(self.data['xs'])
            self.data.close()
            self.data = None
        else:
            _len = len(self.data['xs'])
        return _len

    def __del__(self):
        if self.data is not None:
            self.data.close()



