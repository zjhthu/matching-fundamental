import os

seqs = [
        'te-brown1/',
        #'te-brown2/',
        #'te-brown3/',
        #'te-brown4/',
        #'te-brown5/',
        'te-hotel1/',
        'te-harvard1/',
        #'te-harvard2/',
        #'te-harvard3/',
        #'te-harvard4/',
        'te-mit1/',
        #'te-mit2/',
        #'te-mit3/',
        #'te-mit4/',
        #'te-mit5/'
        ]
data_path_sift='/home/jhzhang/dump_corr_data/data_dump/sift/'
model_sift='../../model/corrnet/sun3d/ours/sift/'

data_path_sp='/home/jhzhang/dump_corr_data/data_dump/sp/'
model_sp='../../model/corrnet/sun3d/ours/superpoint/'

suffix='/numkp-2000/nn-1/nocrop/te-1000'
for seq in seqs:
        #os.mkdir(data_path_sift+seq+suffix+'/result')
        #if not os.path.exists(data_path_sp+seq+suffix+'/result'):
        #    os.mkdir(data_path_sp+seq+suffix+'/result')
        os.system('python main.py --run_mode=test  --kpt_only=True  --use_ransac=False --save_vis=True --vis_geod_th=1e-5'+ \
                ' --data_te='+data_path_sift+seq+suffix + \
                ' --model_path='+model_sift +\
                ' --res_path='+data_path_sift+seq+suffix+'/result'
        )
        os.system('python main.py --run_mode=test  --kpt_only=True  --use_ransac=False --save_vis=True --vis_geod_th=1e-5 '+ \
                ' --data_te='+data_path_sp+seq+suffix + \
                ' --model_path='+model_sp +\
                ' --res_path='+data_path_sp+seq+suffix+'/result'
        )
	#os.system('python main.py --run_mode=test  --kpt_only=True  --use_ransac=False --input_noise=2.0 '+ \
        #        ' --data_te='+data_path_sift+seq+suffix + \
        #        ' --model_path='+model_sift +\
        #        ' --res_path='+data_path_sift+seq+suffix+'/result'
        #)
