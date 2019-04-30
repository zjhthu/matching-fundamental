import torch
import numpy as np
import pickle

# heatmap = torch.rand(2,5,5)
# semi = torch.rand(4)
# window_size = 3
# offset = np.linspace(-(window_size//2),window_size//2,num=(window_size//2)*2+1)
# offset = np.meshgrid(offset,offset)
# offset = torch.from_numpy(np.concatenate((offset[0].reshape(-1,1), offset[1].reshape(-1,1)), axis=1)).long()
# tmp = [[[1,2,2],[1,3,3]], [[2,2,2],[2,3,1]]]
# batch_size = 2

# batch_pts = []
# for batch_idx in range(batch_size):
#     xs, ys = tmp[batch_idx]
#     # return np.zeros((3, 0)), None, None
#     pts = np.zeros((3, len(xs))) # Populate point data sized 3xN.
#     # pts use the coordiante as opencv and grid_sample
#     pts[0, :] = ys
#     pts[1, :] = xs
#     pts[2, :] = heatmap[batch_idx, xs, ys].cpu().numpy()
   
#     #import IPython;IPython.embed()
#     batch_pts.append( np.concatenate((np.ones([1,pts.shape[1]])*batch_idx, pts[:2,:]), axis=0) )
# # need to align the number of kpts when training
# min_pts = np.min(np.asarray([pts.shape[1] for pts in batch_pts]))
# batch_pts = [pts[:,:min_pts] for pts in batch_pts]

# # crop score map 
# batch_pts = torch.from_numpy(np.concatenate(batch_pts,axis=1)).to(semi.device).long()
# region_num = window_size*window_size
# idx_offset = torch.cat([torch.zeros(region_num,1).long(), offset], dim=1).to(semi.device)
# #import pdb;pdb.set_trace()
# # switch from pixel coordiante to matrix coordinate
# batch_pts[[1,2],:] = batch_pts[[2,1],:]
# print(batch_pts)
# region_idx = (batch_pts.view(1, 3, batch_size, min_pts) + idx_offset.view(region_num,3,1,1))
# region_idx = region_idx.permute(1,2,0,3).reshape(3,-1)#3*batch*region_num*min_pts
# #import pdb;pdb.set_trace()
# region = heatmap[region_idx[0],region_idx[1],region_idx[2]].reshape(batch_size,region_num,min_pts) #batch*region_num*min_pts
# print(offset)
# print(heatmap)
# for i in range(len(tmp)):
# 	xs, ys = tmp[i]
# 	for j in range(len(tmp[i][0])):
# 		print(heatmap[i, xs[j]-1:xs[j]+2, ys[j]-1:ys[j]+2])
# 		print(region[i,:,j])

# #batch_pts need to switch back to y,x


#torch.save({'pts':pts,'desc':desc,'heatmap':heatmap}, 'batch.th')
#print('batch pts')
#print(pts.shape)
#print(pts[:,:5,:])
#import pickle
res1 = pickle.load(open('../sample/93143386_3150803750-superpoint.pkl','rb'))
kp1,des1 = res1['npy_kpts'], res1['sift_desc']
res2 = pickle.load(open('../sample/49794379_4567449308-superpoint.pkl','rb'))
kp2,des2 = res2['npy_kpts'], res2['sift_desc']
res4 = pickle.load(open('../sample/65350202_195092272-superpoint.pkl','rb'))
kp4,des4 = res4['npy_kpts'], res4['sift_desc']

pts1 = torch.load('batch_pts1.th')
pts2 = torch.load('batch_pts2.th')

print('kpt1')
print(kp1[:5,:])
print('pts1')
print(pts1[:,0,:5])
print('pts2')
print(pts2[0,:5,:])
print('kpt1 equal?')
print(np.abs(pts2[0,:,:2].detach().cpu().numpy() - kp1[:pts2.shape[1],:]).flatten().max())
print(pts2[0,-1,2])
print(np.abs(pts2[1,:,:2].detach().cpu().numpy() - kp2[:pts2.shape[1],:]).flatten().max())
print(pts2[1,-1,2])
print(np.abs(pts2[3,:,:2].detach().cpu().numpy() - kp4[:pts2.shape[1],:]).flatten().max())
print(pts2[2,-1,2])
#print(np.abs(desc[0,:,:].detach().cpu().numpy().T - des1[:pts.shape[1],:]).flatten().max())
#print(np.abs(pts[1,:,:].detach().cpu().numpy() - kp2[:pts.shape[1],:]).flatten().max())
#print(np.abs(desc[1,:,:].detach().cpu().numpy().T - des2[:pts.shape[1],:]).flatten().max())


