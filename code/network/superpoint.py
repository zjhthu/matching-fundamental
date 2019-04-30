import torch
from torch.nn.parameter import Parameter
import numpy as np
import sys
import time
sys.path.append('../util')
from util import get_pool_result

def nms_fast(in_corners, H, W, dist_thresh):
    """
    Run a faster approximate Non-Max-Suppression on numpy corners shaped:
      3xN [x_i,y_i,conf_i]^T
  
    Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
    are zeros. Iterate through all the 1's and convert them either to -1 or 0.
    Suppress points by setting nearby values to 0.
  
    Grid Value Legend:
    -1 : Kept.
     0 : Empty or suppressed.
     1 : To be processed (converted to either kept or supressed).
  
    NOTE: The NMS first rounds points to integers, so NMS distance might not
    be exactly dist_thresh. It also assumes points are within image boundaries.
  
    Inputs
      in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
      H - Image height.
      W - Image width.
      dist_thresh - Distance to suppress, measured as an infinty norm distance.
    Returns
      nmsed_corners - 3xN numpy matrix with surviving corners.
      nmsed_inds - N length numpy vector with surviving corner indices.
    """
    grid = np.zeros((H, W)).astype(int) # Track NMS data.
    inds = np.zeros((H, W)).astype(int) # Store indices of points.
    # Sort by confidence and round to nearest int.
    inds1 = np.argsort(-in_corners[2,:])
    corners = in_corners[:,inds1]
    rcorners = corners[:2,:].round().astype(int) # Rounded corners.
    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
      return np.zeros((3,0)).astype(int), np.zeros(0).astype(int)
    if rcorners.shape[1] == 1:
      out = np.vstack((rcorners, in_corners[2])).reshape(3,1)
      return out, np.zeros((1)).astype(int)
    # Initialize the grid.
    for i, rc in enumerate(rcorners.T):
      grid[rcorners[1,i], rcorners[0,i]] = 1
      inds[rcorners[1,i], rcorners[0,i]] = i
    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    grid = np.pad(grid, ((pad,pad), (pad,pad)), mode='constant')
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners.T):
      # Account for top and left padding.
      pt = (rc[0]+pad, rc[1]+pad)
      if grid[pt[1], pt[0]] == 1: # If not yet suppressed.
        grid[pt[1]-pad:pt[1]+pad+1, pt[0]-pad:pt[0]+pad+1] = 0
        grid[pt[1], pt[0]] = -1
        count += 1
    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = np.where(grid==-1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = np.argsort(-values)
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]
    return out, out_inds

def spatial_soft_argmax(batch_pts, heatmap, window_size, offset, temp):
    # input:
    # batch_pts: batch_idx, x, y, confidence
    # heatmap: batch*H*W
    batch_size, min_pts = batch_pts.shape[1], batch_pts.shape[2]
    # crop score map 
    region_num = window_size*window_size
    idx_offset = torch.cat([torch.zeros(region_num,1).long(), offset], dim=1).cuda()
    batch_pts[[1,2],:] = batch_pts[[2,1],:]#switch to matrix coordiante, 4*batch_size*min_pts
    region_idx = (batch_pts[:3,:].long().view(1, 3, batch_size, min_pts) + idx_offset.view(region_num,3,1,1))
    region_idx = region_idx.permute(1,2,0,3).reshape(3,-1)#3*batch*region_num*min_pts
    region = heatmap[region_idx[0],region_idx[1],region_idx[2]].reshape(batch_size,region_num,min_pts) #batch*region_num*min_pts
    # TODO: apply gaussian on the window?

    '''
    for batch_idx in range(region.shape[0]):
        for kpt_idx in range(region.shape[2]):
            pts_i = batch_pts[:,batch_idx, kpt_idx]
            for off_idx in range(offset.shape[0]):
                off_pixel = pts_i[1:3].long() + offset[off_idx].cuda()
                region_val = heatmap[batch_idx, off_pixel[0], off_pixel[1]]
                our_val = region[batch_idx, off_idx, kpt_idx]
                assert region_val == our_val
    print('pass check')
    import pdb;pdb.set_trace()
    '''

    # get kpts coordinate with spatial softargmax
    region = region*temp
    region = torch.softmax(region, dim=1)
    soft_offset = (region.unsqueeze(3)*offset.cuda().float().reshape(1,region_num,1,2)).sum(dim=1)#batch*min_pts*2
    #print(soft_offset.max(dim=1))
    batch_pts = batch_pts.reshape(4,batch_size,min_pts).permute(1,2,0)[:,:,1:].float()#batch_size*min_pts*3
    batch_pts[:,:,0:2] = batch_pts[:,:,0:2] + soft_offset 
    # batch_pts need to switch back to y,x
    batch_pts[:,:,[0,1]] = batch_pts[:,:,[1,0]]
    score = batch_pts[:,:,2]
    batch_pts = batch_pts[:,:,:2]
    return batch_pts

class SuperPointNet(torch.nn.Module):
  """ Pytorch definition of SuperPoint Network. """
  def __init__(self):
    super(SuperPointNet, self).__init__()
    self.relu = torch.nn.ReLU(inplace=True)
    self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
    # Shared Encoder.
    self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
    self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
    self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
    self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
    self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
    self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
    self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
    self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
    # Detector Head.
    self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
    self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
    # Descriptor Head.
    self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
    self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

  def forward(self, x):
    """ Forward pass that jointly computes unprocessed point and descriptor
    tensors.
    Input
      x: Image pytorch tensor shaped N x 1 x H x W.
    Output
      semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
      desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
    """
    # Shared Encoder.
    x = self.relu(self.conv1a(x))
    x = self.relu(self.conv1b(x))
    x = self.pool(x)
    x = self.relu(self.conv2a(x))
    x = self.relu(self.conv2b(x))
    x = self.pool(x)
    x = self.relu(self.conv3a(x))
    x = self.relu(self.conv3b(x))
    x = self.pool(x)
    x = self.relu(self.conv4a(x))
    x = self.relu(self.conv4b(x))
    # Detector Head.
    cPa = self.relu(self.convPa(x))
    semi = self.convPb(cPa)
    # Descriptor Head.
    cDa = self.relu(self.convDa(x))
    desc = self.convDb(cDa)
    dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
    desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
    return semi, desc

def depth2space(semi):
    # input: 
    # semi: Bx65xHcxWc
    # apply softmax to score map
    dense = torch.softmax(semi, 1)
    # Remove dustbin.
    nodust = dense[:, :-1, :, :]
    # Reshape to get full resolution heatmap.
    Hc, Wc = dense.shape[2], dense.shape[3]
    cell = int(np.sqrt(nodust.shape[1]))
    H, W = Hc*cell, Wc*cell
    nodust = nodust.permute(0, 2, 3, 1)
    heatmap = torch.reshape(nodust, [-1, Hc, Wc, cell, cell])
    heatmap = heatmap.permute(0, 1, 3, 2, 4)
    heatmap = torch.reshape(heatmap, [-1, H, W])
    return heatmap
    
def get_kpt(args):
    # input: 
    # heatmap: H*W
    # bord: border remove
    # kpt_num: number of kpt
    heatmap, conf_thresh, nms_dist, bord, kpt_num = args
    H, W = heatmap.shape
    xs, ys = np.where(heatmap>conf_thresh)
    if len(xs) == 0:
        print('no keypoint detected in batch '+str(batch_idx))
        print(heatmap[batch_idx].max())
        import pdb;pdb.set_trace()
        return None, None, None
    pts = np.zeros((3, len(xs))) # Populate point data sized 3xN.
    pts[0, :] = ys
    pts[1, :] = xs
    pts[2, :] = heatmap[xs, ys]
    pts, _ = nms_fast(pts, H, W, dist_thresh=nms_dist) # Apply NMS.
    inds = np.argsort(pts[2,:])
    pts = pts[:,inds[::-1]] # Sort by confidence.
    # Remove points along border.
    toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W-bord))
    toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H-bord))
    toremove = np.logical_or(toremoveW, toremoveH)
    pts = pts[:, ~toremove]
    pts = pts[:3,:kpt_num]
    return pts


# class KeypointExtract(object):
class KeypointExtract(torch.nn.Module):
  """ Wrapper around pytorch net to help with pre and post image processing. """
  def __init__(self, config):
    super(KeypointExtract, self).__init__()
    # self.name = 'KeypointExtract'
    self.nms_dist = config.nms_dist
    self.conf_thresh = config.conf_thresh
    self.cell = 8 # Size of each output cell. Keep this fixed.
    self.border_remove = 4 # Remove points this close to the border.
    self.kpt_num = config.num_kp
    self.num_processor = config.num_processor
    self.window_size = config.window_size # window size of spatial softargmax
    self.temp = Parameter(torch.Tensor(1))
    torch.nn.init.constant_(self.temp, config.temp_init)
    # self.temp = temp_init

    offset = np.linspace(-(self.window_size//2),self.window_size//2,num=(self.window_size//2)*2+1)
    offset = np.meshgrid(offset,offset)
    self.offset = torch.from_numpy(np.concatenate((offset[0].reshape(-1,1), offset[1].reshape(-1,1)), axis=1)).long()


  def forward(self, semi, coarse_desc):
    """ Process a numpy image to extract points and descriptors.
    Input
      semi - B*65*Hc*Wc
      coarse_desc - B*C*Hc*Wc
    Output
      batch_pts - BxNx3 numpy array with corners [x_i, y_i, confidence_i]^T.
      desc - Bx256xN numpy array of corresponding unit normalized descriptors.
      heatmap - BxHxW numpy heatmap in range [0,1] of point confidences.
      """
    heatmap = depth2space(semi)
    batch_size, H, W = heatmap.shape

    ''' 
    time2 = time.time()
    pool_arg = []
    for batch_idx in range(batch_size):
        pool_arg += [(heatmap[batch_idx,:,:].detach().cpu().numpy(), self.conf_thresh, \
                            self.nms_dist, self.border_remove, self.kpt_num)]
    pool_res = get_pool_result(self.num_processor, get_kpt, pool_arg)
    print('time 2 '+str(time.time()-time2))
    '''
    #time1 = time.time()
    # # get kpts with nms
    batch_pts = []
    with torch.no_grad():
        candidate = torch.nonzero(heatmap >= self.conf_thresh) # Confidence threshold.
        batch_pts = []
        # may need to be parallel
        for batch_idx in range(batch_size):
            # pts use the coordiante as opencv and grid_sample
            xys = candidate[candidate[:,0] == batch_idx,1:].cpu().numpy()
            xs, ys = xys[:,0], xys[:,1]
            #print(heatmap[batch_idx].max())
            if len(xs) == 0:
                print('no keypoint detected in batch '+str(batch_idx))
                print(heatmap[batch_idx].max())
                import pdb;pdb.set_trace()
                return None, None, None
            # return np.zeros((3, 0)), None, None
            pts = np.zeros((3, len(xs))) # Populate point data sized 3xN.
            pts[0, :] = ys
            pts[1, :] = xs
            pts[2, :] = heatmap[batch_idx, xs, ys].cpu().numpy()
            pts, _ = nms_fast(pts, H, W, dist_thresh=self.nms_dist) # Apply NMS.
            inds = np.argsort(pts[2,:])
            pts = pts[:,inds[::-1]] # Sort by confidence.
            # Remove points along border.
            bord = self.border_remove
            toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W-bord))
            toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H-bord))
            toremove = np.logical_or(toremoveW, toremoveH)
            pts = pts[:, ~toremove]
            pts = pts[:3,:self.kpt_num]
            batch_pts.append(np.concatenate((np.ones([1,pts.shape[1]])*batch_idx, pts), axis=0))

    #print('time 1 '+str(time.time()-time1))
    #batch_pts = []
    #for batch_idx in range(batch_size):
    #    pts = pool_res[batch_idx]
    #    #assert np.array_equal(pts, batch_pts2[batch_idx])
    #    batch_pts.append(np.concatenate((np.ones([1,pts.shape[1]])*batch_idx, pts), axis=0) )
    # need to align the number of kpts when training
    min_pts = np.min(np.asarray([pts.shape[1] for pts in batch_pts]))
    batch_pts = [pts[:,:min_pts] for pts in batch_pts]
    # batch_idx, x, y, confidence
    batch_pts = torch.from_numpy(np.concatenate(batch_pts,axis=1)).cuda().reshape(4,batch_size, min_pts)
    
    #batch_pts_bak = batch_pts.clone()
    #print('temp '+str(self.temp.item()))
    batch_pts = spatial_soft_argmax(batch_pts, heatmap, self.window_size, self.offset, self.temp)# batch*min_pts*2
    #import pdb;pdb.set_trace()
    #diff = batch_pts - batch_pts_bak[1:3,:,:].permute(1,2,0).cuda().float()
    #print(diff.max(dim=1))

    
    # --- Process descriptor.
    D = coarse_desc.shape[1] # B*C*Hc*Wc

    # Interpolate into descriptor map using 2D point locations.
    samp_pts = batch_pts[:,:,[0,1]]
    samp_pts[:,:,0] = (samp_pts[:,:,0] / (float(W)/2.)) - 1.
    samp_pts[:,:,1] = (samp_pts[:,:,1] / (float(H)/2.)) - 1.
    samp_pts = samp_pts.view(batch_size, 1, -1, 2)
    desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts).squeeze(2) # B*C*min_pts
    desc = torch.nn.functional.normalize(desc, dim=1)
    return batch_pts, desc, heatmap
