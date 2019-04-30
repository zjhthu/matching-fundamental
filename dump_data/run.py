import sys
sys.path.append('../util')
from io_util import savepkl
from config import get_config
from dump_data import dump_seq


config, unparsed = get_config()

print('--------DUMP TEST DATA------------')
test_seqs = config.test_seqs.split('.')
for mod in ['test']:
    pairs = []
    max_pair = getattr(config, 'test_max_' + mod[:2]+'_sample')
    for seq in test_seqs:
        pairs += dump_seq(config.dataset_path, seq+'/'+mod, config.vis_th, max_pair)
    savepkl(pairs, config.dump_path+config.dump_prefix+'-'+mod+'-unseen.pkl')

print('--------DUMP TRAIN DATA------------')
with open(config.train_seqs, 'r') as ofp:
    train_seqs = ofp.read().split('\n')
del train_seqs[-1]

for mod in ['train','val','test']:
    pairs = []
    max_pair = getattr(config, 'train_max_' + mod[:2]+'_sample')
    for seq in train_seqs:
        pairs += dump_seq(config.dataset_path, seq+'/'+mod, config.vis_th, max_pair)
    savepkl(pairs, config.dump_path+config.dump_prefix+'-'+mod+'.pkl')

