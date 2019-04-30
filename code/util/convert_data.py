import h5py
import numpy as np
import pickle
import os
import sys
import glob

def convert_data(input_path, output_path):
    pair_idx = 0
    with h5py.File(output_path+'/data.hdf5', 'w') as f:
        xs = f.create_group("xs")
        ys = f.create_group("ys")
        Rs = f.create_group("Rs")
        ts = f.create_group("ts")
        xs_sq = pickle.load(open(glob.glob(input_path+'/xs_*.pkl')[0],'rb'))
        ys_sq = pickle.load(open(glob.glob(input_path+'/ys_*.pkl')[0],'rb'))
        Rs_sq = pickle.load(open(glob.glob(input_path+'/Rs_*.pkl')[0],'rb'))
        ts_sq = pickle.load(open(glob.glob(input_path+'/ts_*.pkl')[0],'rb'))
        seq_len = len(xs_sq)
        for i in range(seq_len):
            xs_i = xs.create_dataset(str(pair_idx), xs_sq[i].shape, dtype=np.float32)
            xs_i[:] = xs_sq[i].astype(np.float32)
            ys_i = ys.create_dataset(str(pair_idx), ys_sq[i].shape, dtype=np.float32)
            ys_i[:] = ys_sq[i].astype(np.float32)
            Rs_i = Rs.create_dataset(str(pair_idx), Rs_sq[i].shape, dtype=np.float32)
            Rs_i[:] = Rs_sq[i].astype(np.float32)
            ts_i = ts.create_dataset(str(pair_idx), ts_sq[i].shape, dtype=np.float32)
            ts_i[:] = ts_sq[i].astype(np.float32)
            pair_idx = pair_idx + 1

if __name__ == "__main__":
    convert_data(sys.argv[1], sys.argv[2])
