import os
import numpy as np

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
data_path_sp='/home/jhzhang/dump_corr_data/data_dump/sp/'
suffix = 'numkp-2000/nn-1/nocrop/te-1000/result/'
output_path='/home/jhzhang/tmp_data/result/'
top_k = 20
for seq in seqs:
        print(seq)
        err_qt_sift = np.loadtxt(data_path_sift+seq+suffix+'/all_acc_qt_auc20_ours.txt')
        err_q_sift = np.loadtxt(data_path_sift+seq+suffix+'/all_acc_q_auc20_ours.txt')
        err_t_sift = np.loadtxt(data_path_sift+seq+suffix+'/all_acc_t_auc20_ours.txt')
        t_gt_sift = np.loadtxt(data_path_sift+seq+suffix+'/all_t_gt.txt')
        t_hat_sift = np.loadtxt(data_path_sift+seq+suffix+'/all_t_hat.txt')

        err_qt_sp = np.loadtxt(data_path_sp+seq+suffix+'/all_acc_qt_auc20_ours.txt')
        err_q_sp = np.loadtxt(data_path_sp+seq+suffix+'/all_acc_q_auc20_ours.txt')
        err_t_sp = np.loadtxt(data_path_sp+seq+suffix+'/all_acc_t_auc20_ours.txt')
        t_gt_sp = np.loadtxt(data_path_sp+seq+suffix+'/all_t_gt.txt')
        t_hat_sp = np.loadtxt(data_path_sp+seq+suffix+'/all_t_hat.txt')

        diff = err_qt_sift - err_qt_sp
        #diff = err_qt_sp - err_qt_sift
        sort_idx = np.argsort(diff)[:top_k]
        sort_val = diff[sort_idx]
        selected_idx = sort_idx[sort_val<0]
        res_dir=output_path+seq
        if not os.path.exists(res_dir):
                os.mkdir(res_dir)
                os.mkdir(res_dir+'/sift')
                os.mkdir(res_dir+'/sp')
        for idx in selected_idx:
                print('cp '+data_path_sift+seq+suffix+str(idx)+'.png'+' \''+res_dir+'/sift/'+str(idx)+'-q-'+str(int(err_q_sift[idx]))+'-t-'+str(int(err_t_sift[idx]))+\
                    '-tgt-'+np.array2string(t_gt_sift[idx], precision=2, separator=',')+'-t-'+np.array2string(t_hat_sift[idx], precision=2, separator=',')+'.png\'')
                #import pdb;pdb.set_trace()
                os.system('cp '+data_path_sift+seq+suffix+str(idx)+'.png'+' \''+res_dir+'/sift/'+str(idx)+'-q-'+str(int(err_q_sift[idx]))+'-t-'+str(int(err_t_sift[idx]))+\
                        '-tgt-'+np.array2string(t_gt_sift[idx], precision=2, separator=',')+'-t-'+np.array2string(t_hat_sift[idx], precision=2, separator=',')+'.png\'')
                os.system('cp '+data_path_sp+seq+suffix+str(idx)+'.png'+' \''+res_dir+'/sp/'+str(idx)+'-q-'+str(int(err_q_sp[idx]))+'-t-'+str(int(err_t_sp[idx]))+\
                        '-tgt-'+np.array2string(t_gt_sp[idx], precision=2, separator=',')+'-t-'+np.array2string(t_hat_sp[idx], precision=2, separator=',')+'.png\'')
                os.system('cp '+data_path_sift+seq+suffix+str(idx)+'-before.png'+' \''+res_dir+'/sift/'+str(idx)+'before-q-'+str(int(err_q_sift[idx]))+'-t-'+str(int(err_t_sift[idx]))+\
                        '-tgt-'+np.array2string(t_gt_sift[idx], precision=2, separator=',')+'-t-'+np.array2string(t_hat_sift[idx], precision=2, separator=',')+'.png\'')
                os.system('cp '+data_path_sp+seq+suffix+str(idx)+'-before.png'+' \''+res_dir+'/sp/'+str(idx)+'before-q-'+str(int(err_q_sp[idx]))+'-t-'+str(int(err_t_sp[idx]))+\
                        '-tgt-'+np.array2string(t_gt_sp[idx], precision=2, separator=',')+'-t-'+np.array2string(t_hat_sp[idx], precision=2, separator=',')+'.png\'')




