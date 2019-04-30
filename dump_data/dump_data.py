from six.moves import xrange
import itertools
import numpy as np
import sys
sys.path.append('../util')
from io_util import loadh5

np.random.seed(1234)

def load_seq(data_path):
    # load sequence information
    print("Preparing data for {}".format(data_path))
    vis_list_file = data_path + "/visibility.txt"
    with open(data_path + "/images.txt",'r') as ofp:
        img_list = ofp.read().split('\n')
    del img_list[-1]
    with open(data_path + "/calibration.txt",'r') as ofp:
        geom_list = ofp.read().split('\n')
    del geom_list[-1]
    with open(data_path + "/visibility.txt",'r') as ofp:
        vis_list = ofp.read().split('\n')
    del vis_list[-1]

    geom = []
    for geom_file in geom_list:
        geom += [loadh5(data_path+'/'+geom_file)]

    vis = []
    for vis_file in vis_list:
        vis += [np.loadtxt(data_path+'/'+vis_file).flatten().astype("float32")]
    vis = np.asarray(vis)
    return img_list, geom, vis

def dump_seq(data_path_base, seq_path, vis_th, max_pair):
    img_list, geom, vis = load_seq(data_path_base+'/'+seq_path)
    pairs = []
    for ii, jj in itertools.product(xrange(len(img_list)), xrange(len(img_list))):
        if vis[ii][jj] > vis_th:
            if ii != jj:
                pairs.append([(ii, jj),(seq_path+'/'+img_list[ii],seq_path+'/'+img_list[jj]), \
                        [geom[ii], geom[jj]]])
    print(str(len(pairs))+ " pairs generated for "+seq_path)
    if len(pairs) > max_pair:
        pairs = [pairs[_i] for _i in np.random.permutation(len(pairs))[:max_pair]]

    return pairs

