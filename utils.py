from asyncio import FastChildWatcher
from cProfile import label
from tkinter.messagebox import NO
from cv2 import RHO
import numpy as np
from pyparsing import col
import torchvision.utils
import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Callable, Union, Dict
import torch.nn.functional as F
# import pdb;pdb.set_trace()
from torch_scatter import scatter_mean
import matplotlib.pyplot as plt
import os
import copy
import open3d as o3d
from typing import Tuple
import cv2

def shuffle_mask(region_mask):
    unique_class=np.unique(region_mask)
    unique_class_new=np.arange(len(unique_class))
    old_class=unique_class.copy()
    np.random.shuffle(unique_class_new)
    region_mask_new=region_mask.copy()
    for old_class_i,old_class_v in enumerate(old_class):
        region_mask_new[region_mask==old_class_v]=unique_class_new[old_class_i]
    return region_mask_new

# project the reference point cloud into the source view, then project back
def reproject_with_depth(
    depth_ref: np.ndarray,
    intrinsics_ref: np.ndarray,
    extrinsics_ref: np.ndarray,
    depth_src: np.ndarray,
    intrinsics_src: np.ndarray,
    extrinsics_src: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project the reference points to the source view, then project back to calculate the reprojection error
    Args:
        depth_ref: depths of points in the reference view, of shape (H, W)
        intrinsics_ref: camera intrinsic of the reference view, of shape (3, 3)
        extrinsics_ref: camera extrinsic of the reference view, of shape (4, 4)
        depth_src: depths of points in the source view, of shape (H, W)
        intrinsics_src: camera intrinsic of the source view, of shape (3, 3)
        extrinsics_src: camera extrinsic of the source view, of shape (4, 4)
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            depth_reprojected: reprojected depths of points in the reference view, of shape (H, W)
            x_reprojected: reprojected x coordinates of points in the reference view, of shape (H, W)
            y_reprojected: reprojected y coordinates of points in the reference view, of shape (H, W)
    """
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    # step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref), np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)), np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    k_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = k_xyz_src[:2] / k_xyz_src[2:3]
    valid_depth1 = (k_xyz_src[2:3]>0).reshape([height, width]) & (depth_ref>0)

    # step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    valid_uv1 = (x_src>=0)&(x_src<=width-1)&(y_src>=0)&(y_src<=height-1)

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src), np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)), np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    k_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = k_xyz_reprojected[:2] / k_xyz_reprojected[2:3]
    valid_depth2 = (k_xyz_reprojected[2:3]>0).reshape([height, width])
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)
    valid_uv2 = (x_reprojected>=0)&(x_reprojected<=width-1)&(y_reprojected>=0)&(y_reprojected<=height-1)
    valid_mask = valid_depth1 & valid_uv1 & valid_depth2 & valid_uv2
    return depth_reprojected, x_reprojected, y_reprojected, valid_mask

def get_neighbors(points):
    """
    args: voxel_coordinates: [b, n_steps, n_samples, 3]
    """
    return torch.stack([
        torch.stack(
            [
                torch.floor(points[:, :, :, 0]),
                torch.floor(points[:, :, :, 1]),
                torch.floor(points[:, :, :, 2])
            ],
            dim=-1
        ),
        torch.stack(
            [
                torch.ceil(points[:, :, :, 0]),
                torch.floor(points[:, :, :, 1]),
                torch.floor(points[:, :, :, 2])
            ],
            dim=-1
        ),
        torch.stack(
            [
                torch.floor(points[:, :, :, 0]),
                torch.ceil(points[:, :, :, 1]),
                torch.floor(points[:, :, :, 2])
            ],
            dim=-1
        ),
        torch.stack(
            [
                torch.floor(points[:, :, :, 0]),
                torch.floor(points[:, :, :, 1]),
                torch.ceil(points[:, :, :, 2])
            ],
            dim=-1
        ),
        torch.stack(
            [
                torch.ceil(points[:, :, :, 0]),
                torch.ceil(points[:, :, :, 1]),
                torch.floor(points[:, :, :, 2])
            ],
            dim=-1
        ),
        torch.stack(
            [
                torch.ceil(points[:, :, :, 0]),
                torch.floor(points[:, :, :, 1]),
                torch.ceil(points[:, :, :, 2])
            ],
            dim=-1
        ),
        torch.stack(
            [
                torch.floor(points[:, :, :, 0]),
                torch.ceil(points[:, :, :, 1]),
                torch.ceil(points[:, :, :, 2])
            ],
            dim=-1
        ),
        torch.stack(
            [
                torch.ceil(points[:, :, :, 0]),
                torch.ceil(points[:, :, :, 1]),
                torch.ceil(points[:, :, :, 2])
            ],
            dim=-1
        ),
    ], dim=1)

def sparse_depth_to_dense(sparse_depth,intrinsics = None, extrinsics = None, images= None, length=3, mindepth=False):
    #* sparse_depth.shape torch.Size([1, 1024, 1280])
    sparse_depth = sparse_depth.half()
    no_depth_mask = sparse_depth<0.001
    sparse_depth[no_depth_mask]=1000.0 #* 零深度转为1000.0
    #* torch.finfo(torch.float16) finfo(resolution=0.001, min=-65504, max=65504, eps=0.000976562, tiny=6.10352e-05, dtype=float16)
    dtype = sparse_depth.dtype
    batch, _, height,width = sparse_depth.shape
    device = sparse_depth.device
    offset_grid = get_offset_grid(length=length)
    offset_grid_torch_list = [torch.tensor(offset, dtype=dtype, device=device) for offset in offset_grid]
    #* 用于采样，返回坐标
    # all_offset_map_torch = torch.stack([rt_up_offset_torch,lt_up_offset_torch,lt_dw_offset_torch,rt_dw_offset_torch],dim=0) #* torch.Size([4, 56, 2])
    all_offset_map_torch = torch.stack(offset_grid_torch_list,dim=0) #* torch.Size([4, 56, 2])
    all_offset_map_torch_batch=all_offset_map_torch.permute(2,0,1).unsqueeze(0).repeat(batch,1,1,1) #* torch.Size([1, 2, 4, 56])
    #* 用于添加偏置
    all_offset_torch = all_offset_map_torch.view(-1,2)
    #* 开始构建图像坐标
    y_grid, x_grid = torch.meshgrid([torch.arange(0, height, dtype=dtype, device=device), torch.arange(0, width, dtype=dtype, device=device),])
    y_grid, x_grid = y_grid.contiguous().view(height * width), x_grid.contiguous().view(height * width)
    xy = torch.stack((x_grid, y_grid))  # [2, H*W]
    xy = torch.unsqueeze(xy, 0).repeat(batch, 1, 1)  # [B, 2, H*W]
    xy_list = []
    #* 为每个点添加偏置坐标
    for i in range(len(all_offset_torch)):
        xy_list.append((xy + all_offset_torch[i].view(2,1)).unsqueeze(2))
    xy_with_offset = torch.cat(xy_list, dim=2)  # [B, 2, 9, H*W]
    x_normalized = xy_with_offset[:, 0, :, :] / ((width - 1) / 2) - 1
    y_normalized = xy_with_offset[:, 1, :, :] / ((height - 1) / 2) - 1
    grid = torch.stack((x_normalized, y_normalized), dim=3)  # [B, 9, H*W, 2]
    # normalized_grid = grid.view(batch, len(all_offset_torch) * height, width, 2)
    # normalized_grid = torch.clamp(normalized_grid, min=-1, max=1)
    #* 归一化坐标采样
    #* surrounding_depth: torch.Size([1, 1, 4, 12, 1024, 1280])
    surrounding_depth = F.grid_sample(sparse_depth, grid, mode="nearest", padding_mode="reflection", align_corners=True).view(batch, -1, 4,len(all_offset_torch)//4, height, width)
    #* 4个区域的最小值及其index
    local_min_v,  local_min_i = torch.min(surrounding_depth,dim=3) #* local_min[0].shape torch.Size([1, 1, 4, 1024, 1280])
    local_min_v = local_min_v.half().squeeze(1)
    #* 利用index还原在all_offset_map_torch_batch下的坐标，方便得到局部坐标
    local_min_uv_u = local_min_i.permute(0,1,3,4,2).half().squeeze(1) #* torch.Size([1, 1, 1024, 1280, 4])
    local_min_uv_v = torch.tensor([i for i in range(4)],dtype=dtype, device=device)[None,None,None,...].repeat(batch,height, width,1)
    local_min_uv_u_norm = local_min_uv_u / ((len(all_offset_torch)//4 - 1) / 2) - 1
    local_min_uv_v_norm = local_min_uv_v / ((4 - 1) / 2) - 1
    local_min_uv_grid = torch.stack([local_min_uv_u_norm,local_min_uv_v_norm],dim=4) #* torch.Size([1, 1024, 1280, 4, 2])
    #* 采样得到局部坐标
    local_min_uv = F.grid_sample(all_offset_map_torch_batch, local_min_uv_grid.view(batch,height,width*4,2), mode="nearest", padding_mode="zeros", align_corners=True)
    local_min_uv = local_min_uv.view(batch, 2, height, width, 4)
    #* 双线性插值得到深度 顺时针排序
    v_up = (local_min_uv[:,1,:,:,0]-local_min_uv[:,1,:,:,3])*(-local_min_uv[:,0,:,:,3])/(local_min_uv[:,0,:,:,0]-local_min_uv[:,0,:,:,3]) + local_min_uv[:,1,:,:,3]
    v_dw = (local_min_uv[:,1,:,:,1]-local_min_uv[:,1,:,:,2])*(-local_min_uv[:,0,:,:,2])/(local_min_uv[:,0,:,:,1]-local_min_uv[:,0,:,:,2]) + local_min_uv[:,1,:,:,2]
    d_up =       (local_min_v[:,0,:,:]-local_min_v[:,3,:,:])*(-local_min_uv[:,0,:,:,3])/(local_min_uv[:,0,:,:,0]-local_min_uv[:,0,:,:,3]) + local_min_v[:,3,:,:]
    d_dw =       (local_min_v[:,1,:,:]-local_min_v[:,2,:,:])*(-local_min_uv[:,0,:,:,2])/(local_min_uv[:,0,:,:,1]-local_min_uv[:,0,:,:,2]) + local_min_v[:,2,:,:]
    d_tmp = (d_dw-d_up)*(-v_up)/(v_dw-v_up) + d_up #* torch.Size([1, 1024, 1280])
    #* local_min_uv.shape torch.Size([1, 2, 1024, 1280, 4])
    local_min_min_v,local_min_min_i = torch.min(local_min_v, dim=1) #* global_min[0].shape torch.Size([1, 1024, 1280])
    local_min_max_v,local_min_max_i = torch.max(local_min_v, dim=1) #* global_min[0].shape torch.Size([1, 1024, 1280])
    local_depth_gap = torch.abs(local_min_max_v-local_min_min_v)
    local_depth_invalid_mask = local_depth_gap>0.5
    d_tmp[local_depth_invalid_mask]=0.0
    d_tmp[d_tmp>40]=0.0
    sparse_depth[sparse_depth>40]=0.0
    comp2sparse = sparse_depth.squeeze(1)*1.0
    sparse2comp = d_tmp*1.0
    # import pdb;pdb.set_trace()
    sparse2comp[sparse2comp==0] = comp2sparse[sparse2comp==0] #* torch.Size([1, 1024, 1280])
    if not mindepth:
        return sparse2comp
    # plot(d_tmp)
    # plot(sparse_depth)
    # plot(comp2sparse)
    # plt.show()
    # import pdb;pdb.set_trace()
    #* 第一次循环属于平滑滤波，基本解决了小的空洞和近似平面区域，但是边界深度变化较大的部分没得到解决，且较大空洞也未解决。
    #* 直接用简单的紧邻法不全其余深度，首先扩大搜索区域
    def min_depth_comp(input_depth,scale = 2):
        smoothing_depth = input_depth*1.0
        invalid_depth_mask = smoothing_depth<=0.0
        smoothing_depth[invalid_depth_mask] = 1000.0 #* 为了寻找局部有效最小深度
        min_depth_filter = smoothing_depth*1.0
        xy_list = []
        #* 为每个点添加偏置坐标
        for i in range(len(all_offset_torch)):
            xy_list.append((xy + scale*all_offset_torch[i].view(2,1)).unsqueeze(2))
        xy_with_offset = torch.cat(xy_list, dim=2)  # [B, 2, 9, H*W]
        x_normalized = xy_with_offset[:, 0, :, :] / ((width - 1) / 2) - 1
        y_normalized = xy_with_offset[:, 1, :, :] / ((height - 1) / 2) - 1
        grid = torch.stack((x_normalized, y_normalized), dim=3)  # [B, 9, H*W, 2]
        surrounding_depth = F.grid_sample(smoothing_depth.unsqueeze(1), grid, mode="nearest", padding_mode="reflection", align_corners=True).view(batch, -1, 4,len(all_offset_torch)//4, height, width) #* torch.Size([1, 1, 4, 12, 1024, 1280])
        #* 4个区域的最小值及其index
        local_min_v,  local_min_i = torch.min(surrounding_depth,dim=3) #* local_min[0].shape torch.Size([1, 1, 4, 1024, 1280])
        local_min_v = local_min_v.half().squeeze(1) #* torch.Size([1, 4, 1024, 1280])
        local_min_min_v,  local_min_min_i = torch.min(local_min_v,dim=1) #* torch.Size([1, 1024, 1280])
        min_depth_filter[invalid_depth_mask] = local_min_min_v[invalid_depth_mask]
        min_depth_filter[min_depth_filter>40.0]=0.0
        output_depth = min_depth_filter*1.0
        # min_depth_filter[min_depth_filter==0] = min_depth_filter.max()*2.0
        # plot(min_depth_filter, name=f'latex/figures/min_depth_filter_length_{length}_scale_{scale}.png')
        return output_depth
    scale = 2
    min_depth_filter = min_depth_comp(sparse2comp,scale)
    num_iter = 0
    while min_depth_filter.min()<=0:
        scale*=2
        min_depth_filter = min_depth_comp(min_depth_filter,scale)
        num_iter+=1
        if num_iter>20:
            break
    # depth2ply(min_depth_filter,images[0].squeeze().permute([1,2,0]),intrinsics[0,0],extrinsics[0,0])
    return min_depth_filter
    import pdb;pdb.set_trace()
    # depth2ply(min_depth_filter,images[0].squeeze().permute([1,2,0]),intrinsics[0,0],extrinsics[0,0])
    comp2sparse[comp2sparse==0] = d_tmp[comp2sparse==0]
    plot(d_tmp)
    plot(sparse_depth)
    plot(comp2sparse)
    #* 绘制
    # sparse2comp[sparse2comp==0] = sparse2comp.max()*2.0
    # plot(sparse2comp, name='latex/figures/sparse2comp.png')
    # sparse_depth[sparse_depth==0] = sparse_depth.max()*2.0
    # plot(sparse_depth,name='latex/figures/sparse_depth.png')
    # plot(d_tmp,name='output/images/d_tmp.png')
    # plot(sparse_depth,name='output/images/sparse_depth.png')
    plt.show()
    import pdb;pdb.set_trace()
    if intrinsics is not None and extrinsics is not None:
        depth2ply(d_tmp,images[0].squeeze().permute([1,2,0]),intrinsics[0,0],extrinsics[0,0])
    import pdb;pdb.set_trace()

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def optimize_depth(images,intrinsics, extrinsics,init_depth,edge_feats,select_ths=0.9, debug=False):
    K = intrinsics[0,0] #* torch.Size([3, 3])
    fx,fy,cx,cy = K[0,0], K[1,1], K[0,2], K[1,2]
    K_inv = torch.inverse(K)
    T_c_w = extrinsics.squeeze(0) #* torch.Size([6, 4, 4]) world to camera 
    T_w_c = torch.inverse(T_c_w) #* torch.Size([6, 4, 4]) camera to world
    depth_samples = init_depth.float()
    depth_samples[depth_samples<=0] = 1.0
    #! 测试2个深度
    # depth_samples = depth_samples.repeat(1,2,1,1)
    _, ndepth, height, width = depth_samples.shape
    batch = edge_feats.shape[0]-1 #* 将一张参考图投影到多张源图 5
    dtype = edge_feats.dtype
    device = edge_feats.device
    #* ===== 提取用于投影的点 ========
    ref_edge = edge_feats[0,0]*1.0
    # debug = True
    if not debug:
        ref_edge_used_mask = ref_edge>select_ths
    else:
        ref_edge_used_mask = ref_edge>=ref_edge.min() #* debug
    # ref_edge_used_mask[0:280,:] = False
    # ref_edge_used_mask[281:,:] = False
    ref_edge_used_mask[:,:164] = False
    ref_edge_used_mask[:,168:] = False
    #* 80 217 184 495
    yy, xx = torch.meshgrid([ torch.arange(0, height, dtype=dtype, device=device), torch.arange(0, width, dtype=dtype, device=device),])
    y, x = yy[ref_edge_used_mask].contiguous().view(-1), xx[ref_edge_used_mask].contiguous().view(-1)
    N = len(x)
    ref_edge_used = ref_edge[ref_edge_used_mask] #* torch.Size([45130])
    depth_all = depth_samples.squeeze(0)
    depth = depth_all[ref_edge_used_mask[None,...].repeat(ndepth,1,1)].view(ndepth,-1) #* torch.Size([2, 45130])
    depth_orig = depth*1.0
    last_depth = depth*1.0
    depth_on_src = []
    #* d*p = K*P_c
    uv1 = torch.stack((x, y, torch.ones_like(x))) #* torch.Size([3, 45130])
    win_offset = []
    win_size, step = 5, 1
    for i in range(0,win_size,step):
        for j in range(0,win_size,step):
            new_offset = [j- win_size//2,i- win_size//2,0]
            if abs(new_offset[0])+abs(new_offset[1])==2 or abs(new_offset[0])+abs(new_offset[1])==0:
                win_offset.append(new_offset)
    all_offset_torch = torch.tensor(win_offset,dtype=dtype, device=device)
    xy_list = []
    #* 为每个点添加偏置坐标
    for i in range(len(all_offset_torch)):
        xy_list.append((uv1 + all_offset_torch[i].view(3,1)).unsqueeze(2))
    xy_with_offset = torch.cat(xy_list, dim=2)
    len_offset = len(win_offset)
    ref_edge_used = ref_edge_used.unsqueeze(1).repeat(1, len_offset).view(-1)
    for iter_n in range(20):
        uvd_offset = xy_with_offset.unsqueeze(0) * depth.unsqueeze(1).unsqueeze(3) #* torch.Size([2, 3, 45130, 9])
        uvd = uvd_offset.view(ndepth,3,-1) #* torch.Size([1, 3, 410868])
        xyz_r = torch.matmul(K_inv,uvd) #* torch.Size([2, 3, 45130]) 参考相机坐标系下的坐标
        #* 投影到世界坐标系 P_w = R_w_c * P_c + t_w_c
        xyz_w = torch.matmul(T_w_c[0,:3,:3],xyz_r) + T_w_c[0,:3,3:] #* torch.Size([2, 3, 45130])
        xyz_w_batch = xyz_w.unsqueeze(0).repeat(batch,1,1,1) #* torch.Size([5, 2, 3, 45130]) [batch, ndepth, 3, N]
        #* 投影到多张源图相机坐标系
        xyz_s_batch = torch.matmul(T_c_w[1:,:3,:3].unsqueeze(1),xyz_w_batch) + T_c_w[1:,:3,3:].unsqueeze(1) #* torch.Size([5, 2, 3, 45130])
        #* 转为像素坐标
        uvd_s_batch = torch.matmul(K,xyz_s_batch)
        #* 错误深度mask
        negative_depth_mask = uvd_s_batch[:,:,2,:] <= 1e-3 #* torch.Size([5, 2, 45130])
        uvd_s_batch[:,:,0,:][negative_depth_mask] = float(width)
        uvd_s_batch[:,:,1,:][negative_depth_mask] = float(height)
        uvd_s_batch[:,:,2,:][negative_depth_mask] = 1.0
        grid_s_uv = uvd_s_batch[:,:,:2:,:] / uvd_s_batch[:,:,2:3,:] #* torch.Size([5, 2, 2, 45130])
        # print("grid_s_uv: ",grid_s_uv)
        grid_s_uv_u_normalized = grid_s_uv[:,:,0,:] / ((width - 1) / 2) - 1
        grid_s_uv_v_normalized = grid_s_uv[:,:,1,:] / ((height - 1) / 2) - 1 #* torch.Size([5, 2, 45130])
        grid_s_uv_normalized = torch.stack((grid_s_uv_u_normalized, grid_s_uv_v_normalized), dim=3) #* torch.Size([5, 2, 45130, 2])
        src_edge_feats_sample = F.grid_sample(edge_feats[1:],grid_s_uv_normalized,mode="bilinear",padding_mode="zeros",align_corners=True,) #* torch.Size([5, 3, 2, 45130])
        #* [batch, 3, ndepth, N]
        #* ======== 开始计算 =======
        #* camera_r to camera_s
        T_cs_cr = torch.matmul(T_c_w[1:], T_w_c[0:1]) #* torch.Size([5, 4, 4])
        t_s_r = T_cs_cr[:,:3,3:] #* torch.Size([5, 3, 1])
        #* 源图相机坐标系下的坐标
        xs,ys,zs = xyz_s_batch[:,:,0], xyz_s_batch[:,:,1], xyz_s_batch[:,:,2]
        #* 逆深度
        rho_r = 1.0 / depth #* torch.Size([2, 45129])
        rho_s = 1.0 / uvd_s_batch[:,:,2,:] #* torch.Size([5, 2, 45129])
        #* 梯度
        bx,by = src_edge_feats_sample[:,1], src_edge_feats_sample[:,2] #* torch.Size([5, 2, 45129])
        #* 边值残差
        residual_edge = src_edge_feats_sample[:,0] - ref_edge_used[None,None,...] #* residual_edge.shape torch.Size([5, 2, 45130])
        # print((residual_edge**2).sum().cpu().numpy())
        #* 对\xi 的导数
        r_xi = torch.zeros_like(residual_edge)[...,None].repeat(1,1,1,6) #* torch.Size([5, 2, 45129, 6])
        #*              b_x * f_x * \rho_s
        r_xi[:,:,:,0] = bx * fx * rho_s
        #*              b_y * f_y * \rho_s
        r_xi[:,:,:,1] = by * fy * rho_s
        #*              -b_x * f_x * x_s * \rho_s^2 - b_y * f_y * y_s * \rho_s^2
        r_xi[:,:,:,2] = -bx * fx * xs * rho_s**2 - by * fy * ys * rho_s**2
        #*              -b_x * f_x * x_s * y_s * \rho_s^2 - b_y * f_y * y_s^2 * \rho_s^2 - b_y * f_y
        r_xi[:,:,:,3] = -bx * fx * xs * ys * rho_s**2 - by * fy * ys**2 * rho_s**2 - by * fy
        #*              b_y * f_y * x_s * y_s * \rho_s^2 + b_x * f_x * x_s^2 * \rho_s^2 + b_x * f_x
        r_xi[:,:,:,4] = by * fy * xs * ys * rho_s**2 + bx * fx * xs**2 * rho_s**2 + bx * fx
        #*              b_y * f_y * x_s * \rho_s - b_x * f_x * y_s * \rho_s
        r_xi[:,:,:,5] = by * fy * xs * rho_s - bx * fx * ys * rho_s
        #* 对\rho_r 的导数
        r_rho_r = torch.zeros_like(residual_edge)[...,None].repeat(1,1,1,1) #* torch.Size([5, 2, 45129, 1])
        #*                  \rho_s / \rho_r * ( b_x*f_x*(t_1 - x_s*\rho_s*t_3) + b_y*f_y*(t_2 - y_s*\rho_s*t_3) )
        # import pdb;pdb.set_trace()
        r_rho_r[:,:,:,0] =  rho_s / rho_r.unsqueeze(2).repeat(1, 1, len_offset).view(ndepth,-1) * (bx * fx * (t_s_r[:,0:1] - xs * rho_s * t_s_r[:,2:]) + by * fy * (t_s_r[:,1:2] - ys * rho_s * t_s_r[:,2:]) )
        r_rho_r[abs(r_rho_r)<10] = 0.0
        #! H 矩阵，分块考虑. 一个点可以影响 batch 个\xi， 但是只能影响一个深度
        #* 左上角
        J_xi_tmp = r_xi.view(batch,-1,1,6)*1.0 #* torch.Size([5, 90260, 6, 1])
        J_xi_tmp_T = r_xi.view(batch,-1,6,1)*1.0 #* torch.Size([5, 90260, 1, 6])
        H_xi_xi_tmp = torch.matmul(J_xi_tmp_T,J_xi_tmp) #* torch.Size([5, 90260, 6, 6]) 一个点的一个深度对于每张源图相机位姿的影响
        H_xi_xi_batch = H_xi_xi_tmp.sum(dim=1) #* torch.Size([5, 6, 6])
        H_xi_xi = torch.zeros((batch*6,batch*6)).type_as(r_xi) #* torch.Size([30, 30])
        for i in range(batch):
            H_xi_xi[6*i:6*(i+1),6*i:6*(i+1)] = H_xi_xi_batch[i]
        #* 右上角
        H_xi_rho_r_tmp = r_rho_r * r_xi #* torch.Size([5, 2, 45130, 6])
        H_xi_rho_r_batch = H_xi_rho_r_tmp.view(batch,-1,6).permute(1,0,2) #* torch.Size([90260, 5, 6])
        H_xi_rho_r = H_xi_rho_r_batch.reshape(-1,batch*6).permute(1,0) #* torch.Size([30, 90260]) 右上角
        # import pdb;pdb.set_trace()
        H_xi_rho_r = H_xi_rho_r.view(batch*6,-1,len_offset).sum(2)
        # import pdb;pdb.set_trace()
        #* 右下角
        H_rho_r_rho_r_tmp = r_rho_r**2 #* torch.Size([5, 2, 45130, 1])
        H_rho_r_rho_r = H_rho_r_rho_r_tmp.view(batch,-1).sum(0) #* torch.Size([90260])
        H_rho_r_rho_r = H_rho_r_rho_r.view(-1,len_offset).sum(1)
        # print(H_rho_r_rho_r.max())
        #! b 矩阵
        #* 位姿相关部分
        b_xi_tmp = r_xi * residual_edge[...,None] #* torch.Size([5, 2, 45130, 6])
        b_xi_tmp_batch = b_xi_tmp.view(batch,-1,6).permute(1,0,2) #* torch.Size([90260, 5, 6])
        b_xi = b_xi_tmp_batch.reshape(-1,batch*6).sum(0) #* torch.Size([30])
        #* 逆深度相关部分
        b_rho_r_tmp = r_rho_r * residual_edge[...,None] #* torch.Size([5, 2, 45130, 1])
        b_rho_r = b_rho_r_tmp.view(batch,-1).sum(0) #* torch.Size([90260])
        b_rho_r = b_rho_r.view(-1,len_offset).sum(1)
        #? 右下角的对角元素为0的位置对应的H矩阵的行列值为0，b对应位置也为0, 这样的话即使将对角元素值设为1，也应该不会影响最终的结果，但是矩阵就可逆了
        # (H_rho_r_rho_r<1e-3).sum()
        H_rho_r_rho_r_orig = H_rho_r_rho_r*1.0
        #* H_rho_r_rho_r_orig.max() tensor(8.5058e-07, device='cuda:0') 
        #* H_rho_r_rho_r_orig.min() 非零最小：tensor(4.4086e-19, device='cuda:0')
        cannot_update = H_rho_r_rho_r == 0
        H_rho_r_rho_r[cannot_update] = 1.0 #* 保证了0变为1不影响结果
        b_rho_r[cannot_update] = 0.0
        # b_rho_r[cannot_update].sum() #* 0
        # H_xi_rho_r[:,cannot_update].sum() #* 0
        #* b_rho_r.min() tensor(-0.0005, device='cuda:0')
        #* b_rho_r.max() tensor(0.0005, device='cuda:0')
        delta_rho_r = -(1.0/H_rho_r_rho_r) * b_rho_r #* torch.Size([45651])
        # import pdb;pdb.set_trace()
        delta_rho_r = delta_rho_r.view(1,-1)
        updated_rho_r = rho_r + delta_rho_r
        updated_rho_r[updated_rho_r<1/40.0] = 1/40.0
        #* delta_rho_r.max() tensor(22606856., device='cuda:0') 
        #* delta_rho_r.min() tensor(-24823658., device='cuda:0')
        # H_rho_r_rho_r_inv = torch.diag_embed(1.0/H_rho_r_rho_r) #* 一个矩阵居然占了7个G
        #* H * \delta x = - b 
        #* H = [ [H_xi_xi,      H_xi_rho_r],
        #*       [H_xi_rho_r^T, diag(H_rho_r_rho_r)] ]
        #* b = [b_xi,
        #*      b_rho_r]
        #* 边缘化你深度，求解位姿变化:
        #* (H_xi_xi - H_xi_rho_r * (H_rho_r_rho_r)^{-1} * H_rho_r_xi)\delat \xi = -(b_xi - H_xi_rho_r * (H_rho_r_rho_r)^{-1} * b_rho_r)
        #* H_rho_r_xi * \delat \xi + H_rho_r_rho_r * \delta \rho_r = -b_rho_r
        last_depth = depth*1.0
        # depth = 1.0/updated_rho_r
        # depth_samples[0][ref_edge_used_mask[None,...].repeat(ndepth,1,1)] = depth.view(-1)
        if True:
        # def plot_depth_result(new_n_depth = 20,delta_d = 0.5):
            yy, xx = torch.meshgrid([ torch.arange(0, height, dtype=dtype, device=device), torch.arange(0, width, dtype=dtype, device=device),])
            y, x = yy[ref_edge_used_mask].contiguous().view(-1), xx[ref_edge_used_mask].contiguous().view(-1)
            ref_edge_used = ref_edge[ref_edge_used_mask] #* torch.Size([45130])
            new_n_depth = 20
            delta_d = 0.5
            d_samples = torch.linspace(1-delta_d,1+delta_d,new_n_depth, dtype=dtype, device=device)
            depth_all = depth_samples.squeeze(0).repeat(new_n_depth,1,1)
            if new_n_depth>1:
                depth_all = depth_all * d_samples[...,None,None]
            depth_pts = depth_all[ref_edge_used_mask[None,...].repeat(new_n_depth,1,1)].view(new_n_depth,-1) #* torch.Size([2, 45130])
            #* d*p = K*P_c
            uv1 = torch.stack((x, y, torch.ones_like(x))) #* torch.Size([3, 45130])
            uvd = uv1.unsqueeze(0) * depth_pts.unsqueeze(1) #* torch.Size([2, 3, 45130])
            xyz_r = torch.matmul(K_inv,uvd) #* torch.Size([2, 3, 45130]) 参考相机坐标系下的坐标
            #* 投影到世界坐标系 P_w = R_w_c * P_c + t_w_c
            xyz_w = torch.matmul(T_w_c[0,:3,:3],xyz_r) + T_w_c[0,:3,3:] #* torch.Size([2, 3, 45130])
            xyz_w_batch = xyz_w.unsqueeze(0).repeat(1,1,1,1) #* torch.Size([5, 2, 3, 45130]) [batch, new_n_depth, 3, N]
            #* 投影到多张源图相机坐标系
            xyz_s_batch = torch.matmul(T_c_w[1:2,:3,:3].unsqueeze(1),xyz_w_batch) + T_c_w[1:2,:3,3:].unsqueeze(1) #* torch.Size([5, 2, 3, 45130])
            #* 转为像素坐标
            uvd_s_batch = torch.matmul(K,xyz_s_batch) #* torch.Size([1, 1, 3, 45652])
            #* 错误深度mask
            negative_depth_mask = uvd_s_batch[:,:,2,:] <= 1e-3 #* torch.Size([5, 2, 45130])
            uvd_s_batch[:,:,0,:][negative_depth_mask] = float(width)
            uvd_s_batch[:,:,1,:][negative_depth_mask] = float(height)
            uvd_s_batch[:,:,2,:][negative_depth_mask] = 1.0
            grid_s_uv = uvd_s_batch[:,:,:2:,:] / uvd_s_batch[:,:,2:3,:] #* torch.Size([1, 5, 2, 2])
            # import pdb;pdb.set_trace()
            grid_s_uv_squeeze = grid_s_uv.permute(0,2,1,3).reshape(2,-1).squeeze().cpu().long().numpy() #* torch.Size([2, 45652])
            grid_s_uv_squeeze_valid_mask = (grid_s_uv_squeeze[0]>0)*(grid_s_uv_squeeze[0]<width-1)*(grid_s_uv_squeeze[1]>0)*(grid_s_uv_squeeze[1]<height-1)
            src_edge = edge_feats[1,0].cpu().numpy()
            zeors = np.zeros_like(src_edge)
            src_edge_g = np.stack((zeors,src_edge,zeors),axis=2)
            x_p = grid_s_uv_squeeze[0][grid_s_uv_squeeze_valid_mask]
            y_p = grid_s_uv_squeeze[1][grid_s_uv_squeeze_valid_mask]
            color_p = np.linspace(0.5,1,new_n_depth)
            color_pts_tmp = np.tile(color_p,len(ref_edge_used))[grid_s_uv_squeeze_valid_mask]
            color_pts = np.stack([color_pts_tmp, np.zeros_like(color_pts_tmp), np.zeros_like(color_pts_tmp)],axis=1)
            if len(y_p) == len(color_pts):
                src_edge_g[y_p,x_p] = color_pts
            else:
                src_edge_g[y_p,x_p] = [1,0,0]
            ref_edge_p = (ref_edge*1.0).squeeze().cpu().numpy()
            ref_edge_g = np.stack((zeors,ref_edge_p,zeors),axis=2) #* (1024, 1280, 3)
            ref_edge_g[ref_edge_used_mask.cpu().numpy(),:]=[1,0,0]
            images_plot = []
            images_plot.append(['ref_edge_g',to_plot_data(ref_edge_g),'jet'])
            images_plot.append(['src_edge_g',to_plot_data(src_edge_g),'jet'])
            if len(depth_on_src)==0:
                depth_on_src.append(['src_edge_g_'+str(iter_n),to_plot_data(src_edge_g),'jet'])
            if len(depth_on_src)>0 and 'src_edge_g_'+str(iter_n) not in depth_on_src[-1][0]:
                depth_on_src.append(['src_edge_g_'+str(iter_n),to_plot_data(src_edge_g),'jet'])
            plt_imgs(images_plot,sharexy=False,show=False)
            plt_imgs(depth_on_src,show=True)
            import pdb;pdb.set_trace()
        # plot_depth_result(new_n_depth = 20,delta_d = 0.5)
        # plot_result()
        import pdb;pdb.set_trace()
    import pdb;pdb.set_trace()
    pass

def optimize_pose(images,intrinsics, extrinsics,init_depth,edge_feats,select_ths=0.9, debug=False):
    K = intrinsics[0,0] #* torch.Size([3, 3])
    fx,fy,cx,cy = K[0,0], K[1,1], K[0,2], K[1,2]
    K_inv = torch.inverse(K)
    T_c_w = extrinsics.squeeze(0) #* torch.Size([6, 4, 4]) world to camera 
    T_w_c = torch.inverse(T_c_w) #* torch.Size([6, 4, 4]) camera to world
    depth_samples = init_depth.float()
    depth_samples[depth_samples<=0] = 1.0
    #! 测试2个深度
    # depth_samples = depth_samples.repeat(1,2,1,1)
    _, ndepth, height, width = depth_samples.shape
    batch = edge_feats.shape[0]-1 #* 将一张参考图投影到多张源图 5
    dtype = edge_feats.dtype
    device = edge_feats.device
    #* ===== 提取用于投影的点 ========
    ref_edge = edge_feats[0,0]*1.0
    # debug = True
    if not debug:
        ref_edge_used_mask = ref_edge>select_ths
    else:
        ref_edge_used_mask = ref_edge>=ref_edge.min() #* debug
    # ref_edge_used_mask[0:281,:] = False
    # ref_edge_used_mask[282:,:] = False
    # ref_edge_used_mask[:,:171] = False
    # ref_edge_used_mask[:,173:] = False
    #* 80 217 184 495
    def plot_edge(select_ths=0.9):
        ref_edge_test = edge_feats[0,0]*1.0
        ref_edge_used_mask = ref_edge_test>select_ths
        images_plot = []
        images_plot.append(['ref_edge_test',to_plot_data(ref_edge_test),'jet'])
        images_plot.append(['ref_edge_used_mask',to_plot_data(ref_edge_used_mask),'jet'])
        plt_imgs(images_plot)
    yy, xx = torch.meshgrid([ torch.arange(0, height, dtype=dtype, device=device), torch.arange(0, width, dtype=dtype, device=device),])
    y, x = yy[ref_edge_used_mask].contiguous().view(-1), xx[ref_edge_used_mask].contiguous().view(-1)
    N = len(x)
    ref_edge_used = ref_edge[ref_edge_used_mask] #* torch.Size([45130])
    depth_all = depth_samples.squeeze(0)
    depth = depth_all[ref_edge_used_mask[None,...].repeat(ndepth,1,1)].view(ndepth,-1) #* torch.Size([2, 45130])
    depth_orig = depth*1.0
    last_depth = depth*1.0
    depth_on_src = []
    #* d*p = K*P_c
    uv1 = torch.stack((x, y, torch.ones_like(x))) #* torch.Size([3, 45130])
    win_offset = []
    win_size, step = 5, 1
    for i in range(0,win_size,step):
        for j in range(0,win_size,step):
            new_offset = [j- win_size//2,i- win_size//2,0]
            if abs(new_offset[0])+abs(new_offset[1])==2 or abs(new_offset[0])+abs(new_offset[1])==0:
                win_offset.append(new_offset)
    all_offset_torch = torch.tensor(win_offset,dtype=dtype, device=device)
    xy_list = []
    #* 为每个点添加偏置坐标
    for i in range(len(all_offset_torch)):
        xy_list.append((uv1 + all_offset_torch[i].view(3,1)).unsqueeze(2))
    xy_with_offset = torch.cat(xy_list, dim=2)
    len_offset = len(win_offset)
    ref_edge_used = ref_edge_used.unsqueeze(1).repeat(1, len_offset).view(-1)
    for iter_n in range(4):
        uvd_offset = xy_with_offset.unsqueeze(0) * depth.unsqueeze(1).unsqueeze(3) #* torch.Size([2, 3, 45130, 9])
        uvd = uvd_offset.view(ndepth,3,-1) #* torch.Size([1, 3, 410868])
        xyz_r = torch.matmul(K_inv,uvd) #* torch.Size([2, 3, 45130]) 参考相机坐标系下的坐标
        #* 投影到世界坐标系 P_w = R_w_c * P_c + t_w_c
        xyz_w = torch.matmul(T_w_c[0,:3,:3],xyz_r) + T_w_c[0,:3,3:] #* torch.Size([2, 3, 45130])
        xyz_w_batch = xyz_w.unsqueeze(0).repeat(batch,1,1,1) #* torch.Size([5, 2, 3, 45130]) [batch, ndepth, 3, N]
        #* 投影到多张源图相机坐标系
        xyz_s_batch = torch.matmul(T_c_w[1:,:3,:3].unsqueeze(1),xyz_w_batch) + T_c_w[1:,:3,3:].unsqueeze(1) #* torch.Size([5, 2, 3, 45130])
        #* 转为像素坐标
        uvd_s_batch = torch.matmul(K,xyz_s_batch)
        #* 错误深度mask
        negative_depth_mask = uvd_s_batch[:,:,2,:] <= 1e-3 #* torch.Size([5, 2, 45130])
        uvd_s_batch[:,:,0,:][negative_depth_mask] = float(width)
        uvd_s_batch[:,:,1,:][negative_depth_mask] = float(height)
        uvd_s_batch[:,:,2,:][negative_depth_mask] = 1.0
        grid_s_uv = uvd_s_batch[:,:,:2:,:] / uvd_s_batch[:,:,2:3,:] #* torch.Size([5, 2, 2, 45130])
        # print("grid_s_uv: ",grid_s_uv)
        grid_s_uv_u_normalized = grid_s_uv[:,:,0,:] / ((width - 1) / 2) - 1
        grid_s_uv_v_normalized = grid_s_uv[:,:,1,:] / ((height - 1) / 2) - 1 #* torch.Size([5, 2, 45130])
        grid_s_uv_normalized = torch.stack((grid_s_uv_u_normalized, grid_s_uv_v_normalized), dim=3) #* torch.Size([5, 2, 45130, 2])
        src_edge_feats_sample = F.grid_sample(edge_feats[1:],grid_s_uv_normalized,mode="bilinear",padding_mode="zeros",align_corners=True,) #* torch.Size([5, 3, 2, 45130])
        #* [batch, 3, ndepth, N]
        #* ======== 开始计算 =======
        #* camera_r to camera_s
        T_cs_cr = torch.matmul(T_c_w[1:], T_w_c[0:1]) #* torch.Size([5, 4, 4])
        t_s_r = T_cs_cr[:,:3,3:] #* torch.Size([5, 3, 1])
        #* 源图相机坐标系下的坐标
        xs,ys,zs = xyz_s_batch[:,:,0], xyz_s_batch[:,:,1], xyz_s_batch[:,:,2]
        #* 逆深度
        rho_r = 1.0 / depth #* torch.Size([2, 45129])
        rho_s = 1.0 / uvd_s_batch[:,:,2,:] #* torch.Size([5, 2, 45129])
        #* 梯度
        bx,by = src_edge_feats_sample[:,1], src_edge_feats_sample[:,2] #* torch.Size([5, 2, 45129])
        #* 边值残差
        residual_edge = src_edge_feats_sample[:,0] - ref_edge_used[None,None,...] #* residual_edge.shape torch.Size([5, 2, 45130])
        # print((residual_edge**2).sum().cpu().numpy())
        #* 对\xi 的导数
        r_xi = torch.zeros_like(residual_edge)[...,None].repeat(1,1,1,6) #* torch.Size([5, 2, 45129, 6])
        #*              b_x * f_x * \rho_s
        r_xi[:,:,:,0] = bx * fx * rho_s
        #*              b_y * f_y * \rho_s
        r_xi[:,:,:,1] = by * fy * rho_s
        #*              -b_x * f_x * x_s * \rho_s^2 - b_y * f_y * y_s * \rho_s^2
        r_xi[:,:,:,2] = -bx * fx * xs * rho_s**2 - by * fy * ys * rho_s**2
        #*              -b_x * f_x * x_s * y_s * \rho_s^2 - b_y * f_y * y_s^2 * \rho_s^2 - b_y * f_y
        r_xi[:,:,:,3] = -bx * fx * xs * ys * rho_s**2 - by * fy * ys**2 * rho_s**2 - by * fy
        #*              b_y * f_y * x_s * y_s * \rho_s^2 + b_x * f_x * x_s^2 * \rho_s^2 + b_x * f_x
        r_xi[:,:,:,4] = by * fy * xs * ys * rho_s**2 + bx * fx * xs**2 * rho_s**2 + bx * fx
        #*              b_y * f_y * x_s * \rho_s - b_x * f_x * y_s * \rho_s
        r_xi[:,:,:,5] = by * fy * xs * rho_s - bx * fx * ys * rho_s
        #* 对\rho_r 的导数
        r_rho_r = torch.zeros_like(residual_edge)[...,None].repeat(1,1,1,1) #* torch.Size([5, 2, 45129, 1])
        #*                  \rho_s / \rho_r * ( b_x*f_x*(t_1 - x_s*\rho_s*t_3) + b_y*f_y*(t_2 - y_s*\rho_s*t_3) )
        # import pdb;pdb.set_trace()
        r_rho_r[:,:,:,0] =  rho_s / rho_r.unsqueeze(2).repeat(1, 1, len_offset).view(ndepth,-1) * (bx * fx * (t_s_r[:,0:1] - xs * rho_s * t_s_r[:,2:]) + by * fy * (t_s_r[:,1:2] - ys * rho_s * t_s_r[:,2:]) )
        r_rho_r[abs(r_rho_r)<10] = 0.0
        #! H 矩阵，分块考虑. 一个点可以影响 batch 个\xi， 但是只能影响一个深度
        #* 左上角
        J_xi_tmp = r_xi.view(batch,-1,1,6)*1.0 #* torch.Size([5, 90260, 6, 1])
        J_xi_tmp_T = r_xi.view(batch,-1,6,1)*1.0 #* torch.Size([5, 90260, 1, 6])
        H_xi_xi_tmp = torch.matmul(J_xi_tmp_T,J_xi_tmp) #* torch.Size([5, 90260, 6, 6]) 一个点的一个深度对于每张源图相机位姿的影响
        H_xi_xi_batch = H_xi_xi_tmp.sum(dim=1) #* torch.Size([5, 6, 6])
        H_xi_xi = torch.zeros((batch*6,batch*6)).type_as(r_xi) #* torch.Size([30, 30])
        for i in range(batch):
            H_xi_xi[6*i:6*(i+1),6*i:6*(i+1)] = H_xi_xi_batch[i]
        #* 右上角
        H_xi_rho_r_tmp = r_rho_r * r_xi #* torch.Size([5, 2, 45130, 6])
        H_xi_rho_r_batch = H_xi_rho_r_tmp.view(batch,-1,6).permute(1,0,2) #* torch.Size([90260, 5, 6])
        H_xi_rho_r = H_xi_rho_r_batch.reshape(-1,batch*6).permute(1,0) #* torch.Size([30, 90260]) 右上角
        # import pdb;pdb.set_trace()
        H_xi_rho_r = H_xi_rho_r.view(batch*6,-1,len_offset).sum(2)
        # import pdb;pdb.set_trace()
        #* 右下角
        H_rho_r_rho_r_tmp = r_rho_r**2 #* torch.Size([5, 2, 45130, 1])
        H_rho_r_rho_r = H_rho_r_rho_r_tmp.view(batch,-1).sum(0) #* torch.Size([90260])
        H_rho_r_rho_r = H_rho_r_rho_r.view(-1,len_offset).sum(1)
        # print(H_rho_r_rho_r.max())
        #! b 矩阵
        #* 位姿相关部分
        b_xi_tmp = r_xi * residual_edge[...,None] #* torch.Size([5, 2, 45130, 6])
        b_xi_tmp_batch = b_xi_tmp.view(batch,-1,6).permute(1,0,2) #* torch.Size([90260, 5, 6])
        b_xi = b_xi_tmp_batch.reshape(-1,batch*6).sum(0) #* torch.Size([30])
        #* 逆深度相关部分
        b_rho_r_tmp = r_rho_r * residual_edge[...,None] #* torch.Size([5, 2, 45130, 1])
        b_rho_r = b_rho_r_tmp.view(batch,-1).sum(0) #* torch.Size([90260])
        b_rho_r = b_rho_r.view(-1,len_offset).sum(1)
        #? 右下角的对角元素为0的位置对应的H矩阵的行列值为0，b对应位置也为0, 这样的话即使将对角元素值设为1，也应该不会影响最终的结果，但是矩阵就可逆了
        # (H_rho_r_rho_r<1e-3).sum()
        H_rho_r_rho_r_orig = H_rho_r_rho_r*1.0
        #* H_rho_r_rho_r_orig.max() tensor(8.5058e-07, device='cuda:0') 
        #* H_rho_r_rho_r_orig.min() 非零最小：tensor(4.4086e-19, device='cuda:0')
        cannot_update = H_rho_r_rho_r == 0
        H_rho_r_rho_r[cannot_update] = 1.0 #* 保证了0变为1不影响结果
        b_rho_r[cannot_update] = 0.0
        # b_rho_r[cannot_update].sum() #* 0
        # H_xi_rho_r[:,cannot_update].sum() #* 0
        #* b_rho_r.min() tensor(-0.0005, device='cuda:0')
        #* b_rho_r.max() tensor(0.0005, device='cuda:0')
        delta_rho_r = -(1.0/H_rho_r_rho_r) * b_rho_r #* torch.Size([45651])
        # import pdb;pdb.set_trace()
        delta_rho_r = delta_rho_r.view(1,-1)
        # print('H_rho_r_rho_r: ',H_rho_r_rho_r)
        # print('b_rho_r: ',b_rho_r)
        # print('delta_rho_r: ',delta_rho_r)
        updated_rho_r = rho_r + delta_rho_r
        updated_rho_r[updated_rho_r<1/40.0] = 1/40.0
        #* delta_rho_r.max() tensor(22606856., device='cuda:0') 
        #* delta_rho_r.min() tensor(-24823658., device='cuda:0')
        invalid_mask = abs(delta_rho_r)>0.5
        # import pdb;pdb.set_trace()
        #* H * \delta x = - b 
        #* H = [ [H_xi_xi,      H_xi_rho_r],
        #*       [H_xi_rho_r^T, diag(H_rho_r_rho_r)] ]
        #* b = [b_xi,
        #*      b_rho_r]
        #* 边缘化你深度，求解位姿变化:
        #* (H_xi_xi - H_xi_rho_r * (H_rho_r_rho_r)^{-1} * H_rho_r_xi)\delat \xi = -(b_xi - H_xi_rho_r * (H_rho_r_rho_r)^{-1} * b_rho_r)
        #* H_rho_r_xi * \delat \xi + H_rho_r_rho_r * \delta \rho_r = -b_rho_r
        H_schur = H_xi_xi - (H_xi_rho_r * (1.0/H_rho_r_rho_r).unsqueeze(0) ) @ H_xi_rho_r.T
        b_schur = b_xi - (H_xi_rho_r * (1.0/H_rho_r_rho_r).unsqueeze(0) ) @ b_rho_r
        # delta_xi = torch.linalg.solve(H_xi_xi, - b_xi).reshape(batch,6)
        delta_xi = torch.linalg.solve(H_schur, - b_schur).reshape(batch,6)
        # torch.allclose(b_schur,H_schur @ delta_xi) #* True
        delta_T = xi2T(delta_xi) #* torch.Size([1, 4, 4])
        T_c_w[1:] = torch.matmul(delta_T, T_c_w[1:])
        def plot_depth_result(new_n_depth = 20,delta_d = 0.5):
            yy, xx = torch.meshgrid([ torch.arange(0, height, dtype=dtype, device=device), torch.arange(0, width, dtype=dtype, device=device),])
            y, x = yy[ref_edge_used_mask].contiguous().view(-1), xx[ref_edge_used_mask].contiguous().view(-1)
            ref_edge_used = ref_edge[ref_edge_used_mask] #* torch.Size([45130])
            # new_n_depth = 1
            # delta_d = 0
            d_samples = torch.linspace(1-delta_d,1+delta_d,new_n_depth, dtype=dtype, device=device)
            depth_all = depth_samples.squeeze(0).repeat(new_n_depth,1,1)
            if new_n_depth>1:
                depth_all = depth_all * d_samples[...,None,None]
            depth = depth_all[ref_edge_used_mask[None,...].repeat(new_n_depth,1,1)].view(new_n_depth,-1) #* torch.Size([2, 45130])
            #* d*p = K*P_c
            uv1 = torch.stack((x, y, torch.ones_like(x))) #* torch.Size([3, 45130])
            uvd = uv1.unsqueeze(0) * depth.unsqueeze(1) #* torch.Size([2, 3, 45130])
            xyz_r = torch.matmul(K_inv,uvd) #* torch.Size([2, 3, 45130]) 参考相机坐标系下的坐标
            #* 投影到世界坐标系 P_w = R_w_c * P_c + t_w_c
            xyz_w = torch.matmul(T_w_c[0,:3,:3],xyz_r) + T_w_c[0,:3,3:] #* torch.Size([2, 3, 45130])
            xyz_w_batch = xyz_w.unsqueeze(0).repeat(1,1,1,1) #* torch.Size([5, 2, 3, 45130]) [batch, new_n_depth, 3, N]
            #* 投影到多张源图相机坐标系
            xyz_s_batch = torch.matmul(T_c_w[1:2,:3,:3].unsqueeze(1),xyz_w_batch) + T_c_w[1:2,:3,3:].unsqueeze(1) #* torch.Size([5, 2, 3, 45130])
            #* 转为像素坐标
            uvd_s_batch = torch.matmul(K,xyz_s_batch) #* torch.Size([1, 1, 3, 45652])
            #* 错误深度mask
            negative_depth_mask = uvd_s_batch[:,:,2,:] <= 1e-3 #* torch.Size([5, 2, 45130])
            uvd_s_batch[:,:,0,:][negative_depth_mask] = float(width)
            uvd_s_batch[:,:,1,:][negative_depth_mask] = float(height)
            uvd_s_batch[:,:,2,:][negative_depth_mask] = 1.0
            grid_s_uv = uvd_s_batch[:,:,:2:,:] / uvd_s_batch[:,:,2:3,:] #* torch.Size([1, 5, 2, 2])
            # import pdb;pdb.set_trace()
            grid_s_uv_squeeze = grid_s_uv.permute(0,2,1,3).reshape(2,-1).squeeze().cpu().long().numpy() #* torch.Size([2, 45652])
            grid_s_uv_squeeze_valid_mask = (grid_s_uv_squeeze[0]>0)*(grid_s_uv_squeeze[0]<width-1)*(grid_s_uv_squeeze[1]>0)*(grid_s_uv_squeeze[1]<height-1)
            src_edge = edge_feats[1,0].cpu().numpy()
            zeors = np.zeros_like(src_edge)
            src_edge_g = np.stack((zeors,src_edge,zeors),axis=2)
            x_p = grid_s_uv_squeeze[0][grid_s_uv_squeeze_valid_mask]
            y_p = grid_s_uv_squeeze[1][grid_s_uv_squeeze_valid_mask]
            src_edge_g[y_p,x_p] = [1,0,0]
            ref_edge_p = (ref_edge*1.0).squeeze().cpu().numpy()
            ref_edge_g = np.stack((zeors,ref_edge_p,zeors),axis=2) #* (1024, 1280, 3)
            ref_edge_g[ref_edge_used_mask.cpu().numpy(),:]=[1,0,0]
            images_plot = []
            images_plot.append(['ref_edge_g',to_plot_data(ref_edge_g),'jet'])
            images_plot.append(['src_edge_g',to_plot_data(src_edge_g),'jet'])
            if len(depth_on_src)==0:
                depth_on_src.append(['src_edge_g_'+str(iter_n),to_plot_data(src_edge_g),'jet'])
            if len(depth_on_src)>0 and 'src_edge_g_'+str(iter_n) not in depth_on_src[-1][0]:
                depth_on_src.append(['src_edge_g_'+str(iter_n),to_plot_data(src_edge_g),'jet'])
            plt_imgs(images_plot,sharexy=False,show=False)
            plt_imgs(depth_on_src,show=True)
            # import pdb;pdb.set_trace()
        # plot_depth_result(new_n_depth = 1,delta_d = 0)
        # import pdb;pdb.set_trace()
    # T_c_w = extrinsics.squeeze(0)
    return T_c_w[None,...]

def optimize_pose_and_depth(images,intrinsics, extrinsics,init_depth,edge_feats,select_ths=0.9, debug=False):
    K = intrinsics[0,0] #* torch.Size([3, 3])
    fx,fy,cx,cy = K[0,0], K[1,1], K[0,2], K[1,2]
    K_inv = torch.inverse(K)
    T_c_w = extrinsics.squeeze(0) #* torch.Size([6, 4, 4]) world to camera 
    T_w_c = torch.inverse(T_c_w) #* torch.Size([6, 4, 4]) camera to world
    depth_samples = init_depth.float()
    depth_samples[depth_samples<=0] = 1.0
    #! 测试2个深度
    # depth_samples = depth_samples.repeat(1,2,1,1)
    _, ndepth, height, width = depth_samples.shape
    batch = edge_feats.shape[0]-1 #* 将一张参考图投影到多张源图 5
    dtype = edge_feats.dtype
    device = edge_feats.device
    #* ===== 提取用于投影的点 ========
    ref_edge = edge_feats[0,0]*1.0
    # debug = True
    if not debug:
        ref_edge_used_mask = ref_edge>select_ths
    else:
        ref_edge_used_mask = ref_edge>=ref_edge.min() #* debug
    # ref_edge_used_mask[0:281,:] = False
    # ref_edge_used_mask[282:,:] = False
    # ref_edge_used_mask[:,:171] = False
    # ref_edge_used_mask[:,173:] = False
    #* 80 217 184 495
    def plot_edge(select_ths=0.9):
        ref_edge_test = edge_feats[0,0]*1.0
        ref_edge_used_mask = ref_edge_test>select_ths
        images_plot = []
        images_plot.append(['ref_edge_test',to_plot_data(ref_edge_test),'jet'])
        images_plot.append(['ref_edge_used_mask',to_plot_data(ref_edge_used_mask),'jet'])
        plt_imgs(images_plot)
    yy, xx = torch.meshgrid([ torch.arange(0, height, dtype=dtype, device=device), torch.arange(0, width, dtype=dtype, device=device),])
    y, x = yy[ref_edge_used_mask].contiguous().view(-1), xx[ref_edge_used_mask].contiguous().view(-1)
    N = len(x)
    ref_edge_used = ref_edge[ref_edge_used_mask] #* torch.Size([45130])
    depth_all = depth_samples.squeeze(0)
    depth = depth_all[ref_edge_used_mask[None,...].repeat(ndepth,1,1)].view(ndepth,-1) #* torch.Size([2, 45130])
    depth_orig = depth*1.0
    last_depth = depth*1.0
    depth_on_src = []
    #* d*p = K*P_c
    uv1 = torch.stack((x, y, torch.ones_like(x))) #* torch.Size([3, 45130])
    win_offset = []
    win_size, step = 5, 1
    for i in range(0,win_size,step):
        for j in range(0,win_size,step):
            new_offset = [j- win_size//2,i- win_size//2,0]
            if abs(new_offset[0])+abs(new_offset[1])==2 or abs(new_offset[0])+abs(new_offset[1])==0:
                win_offset.append(new_offset)
    all_offset_torch = torch.tensor(win_offset,dtype=dtype, device=device)
    xy_list = []
    #* 为每个点添加偏置坐标
    for i in range(len(all_offset_torch)):
        xy_list.append((uv1 + all_offset_torch[i].view(3,1)).unsqueeze(2))
    xy_with_offset = torch.cat(xy_list, dim=2)
    len_offset = len(win_offset)
    ref_edge_used = ref_edge_used.unsqueeze(1).repeat(1, len_offset).view(-1)
    for iter_n in range(20):
        uvd_offset = xy_with_offset.unsqueeze(0) * depth.unsqueeze(1).unsqueeze(3) #* torch.Size([2, 3, 45130, 9])
        uvd = uvd_offset.view(ndepth,3,-1) #* torch.Size([1, 3, 410868])
        xyz_r = torch.matmul(K_inv,uvd) #* torch.Size([2, 3, 45130]) 参考相机坐标系下的坐标
        #* 投影到世界坐标系 P_w = R_w_c * P_c + t_w_c
        xyz_w = torch.matmul(T_w_c[0,:3,:3],xyz_r) + T_w_c[0,:3,3:] #* torch.Size([2, 3, 45130])
        xyz_w_batch = xyz_w.unsqueeze(0).repeat(batch,1,1,1) #* torch.Size([5, 2, 3, 45130]) [batch, ndepth, 3, N]
        #* 投影到多张源图相机坐标系
        xyz_s_batch = torch.matmul(T_c_w[1:,:3,:3].unsqueeze(1),xyz_w_batch) + T_c_w[1:,:3,3:].unsqueeze(1) #* torch.Size([5, 2, 3, 45130])
        #* 转为像素坐标
        uvd_s_batch = torch.matmul(K,xyz_s_batch)
        #* 错误深度mask
        negative_depth_mask = uvd_s_batch[:,:,2,:] <= 1e-3 #* torch.Size([5, 2, 45130])
        uvd_s_batch[:,:,0,:][negative_depth_mask] = float(width)
        uvd_s_batch[:,:,1,:][negative_depth_mask] = float(height)
        uvd_s_batch[:,:,2,:][negative_depth_mask] = 1.0
        grid_s_uv = uvd_s_batch[:,:,:2:,:] / uvd_s_batch[:,:,2:3,:] #* torch.Size([5, 2, 2, 45130])
        # print("grid_s_uv: ",grid_s_uv)
        grid_s_uv_u_normalized = grid_s_uv[:,:,0,:] / ((width - 1) / 2) - 1
        grid_s_uv_v_normalized = grid_s_uv[:,:,1,:] / ((height - 1) / 2) - 1 #* torch.Size([5, 2, 45130])
        grid_s_uv_normalized = torch.stack((grid_s_uv_u_normalized, grid_s_uv_v_normalized), dim=3) #* torch.Size([5, 2, 45130, 2])
        src_edge_feats_sample = F.grid_sample(edge_feats[1:],grid_s_uv_normalized,mode="bilinear",padding_mode="zeros",align_corners=True,) #* torch.Size([5, 3, 2, 45130])
        #* [batch, 3, ndepth, N]
        #* ======== 开始计算 =======
        #* camera_r to camera_s
        T_cs_cr = torch.matmul(T_c_w[1:], T_w_c[0:1]) #* torch.Size([5, 4, 4])
        t_s_r = T_cs_cr[:,:3,3:] #* torch.Size([5, 3, 1])
        #* 源图相机坐标系下的坐标
        xs,ys,zs = xyz_s_batch[:,:,0], xyz_s_batch[:,:,1], xyz_s_batch[:,:,2]
        #* 逆深度
        rho_r = 1.0 / depth #* torch.Size([2, 45129])
        rho_s = 1.0 / uvd_s_batch[:,:,2,:] #* torch.Size([5, 2, 45129])
        #* 梯度
        bx,by = src_edge_feats_sample[:,1], src_edge_feats_sample[:,2] #* torch.Size([5, 2, 45129])
        #* 边值残差
        residual_edge = src_edge_feats_sample[:,0] - ref_edge_used[None,None,...] #* residual_edge.shape torch.Size([5, 2, 45130])
        # print((residual_edge**2).sum().cpu().numpy())
        #* 对\xi 的导数
        r_xi = torch.zeros_like(residual_edge)[...,None].repeat(1,1,1,6) #* torch.Size([5, 2, 45129, 6])
        #*              b_x * f_x * \rho_s
        r_xi[:,:,:,0] = bx * fx * rho_s
        #*              b_y * f_y * \rho_s
        r_xi[:,:,:,1] = by * fy * rho_s
        #*              -b_x * f_x * x_s * \rho_s^2 - b_y * f_y * y_s * \rho_s^2
        r_xi[:,:,:,2] = -bx * fx * xs * rho_s**2 - by * fy * ys * rho_s**2
        #*              -b_x * f_x * x_s * y_s * \rho_s^2 - b_y * f_y * y_s^2 * \rho_s^2 - b_y * f_y
        r_xi[:,:,:,3] = -bx * fx * xs * ys * rho_s**2 - by * fy * ys**2 * rho_s**2 - by * fy
        #*              b_y * f_y * x_s * y_s * \rho_s^2 + b_x * f_x * x_s^2 * \rho_s^2 + b_x * f_x
        r_xi[:,:,:,4] = by * fy * xs * ys * rho_s**2 + bx * fx * xs**2 * rho_s**2 + bx * fx
        #*              b_y * f_y * x_s * \rho_s - b_x * f_x * y_s * \rho_s
        r_xi[:,:,:,5] = by * fy * xs * rho_s - bx * fx * ys * rho_s
        #* 对\rho_r 的导数
        r_rho_r = torch.zeros_like(residual_edge)[...,None].repeat(1,1,1,1) #* torch.Size([5, 2, 45129, 1])
        #*                  \rho_s / \rho_r * ( b_x*f_x*(t_1 - x_s*\rho_s*t_3) + b_y*f_y*(t_2 - y_s*\rho_s*t_3) )
        # import pdb;pdb.set_trace()
        r_rho_r[:,:,:,0] =  rho_s / rho_r.unsqueeze(2).repeat(1, 1, len_offset).view(ndepth,-1) * (bx * fx * (t_s_r[:,0:1] - xs * rho_s * t_s_r[:,2:]) + by * fy * (t_s_r[:,1:2] - ys * rho_s * t_s_r[:,2:]) )
        # print("rho_s / rho_r: ",rho_s / rho_r)
        # print('bx * fx * (t_s_r[:,0:1] - xs * rho_s * t_s_r[:,2:]): ',bx * fx * (t_s_r[:,0:1] - xs * rho_s * t_s_r[:,2:]))
        # print('by * fy * (t_s_r[:,1:2] - ys * rho_s * t_s_r[:,2:]): ',by * fy * (t_s_r[:,1:2] - ys * rho_s * t_s_r[:,2:]))
        r_rho_r[abs(r_rho_r)<10] = 0.0
        #! H 矩阵，分块考虑. 一个点可以影响 batch 个\xi， 但是只能影响一个深度
        #* 左上角
        J_xi_tmp = r_xi.view(batch,-1,1,6)*1.0 #* torch.Size([5, 90260, 6, 1])
        J_xi_tmp_T = r_xi.view(batch,-1,6,1)*1.0 #* torch.Size([5, 90260, 1, 6])
        H_xi_xi_tmp = torch.matmul(J_xi_tmp_T,J_xi_tmp) #* torch.Size([5, 90260, 6, 6]) 一个点的一个深度对于每张源图相机位姿的影响
        H_xi_xi_batch = H_xi_xi_tmp.sum(dim=1) #* torch.Size([5, 6, 6])
        H_xi_xi = torch.zeros((batch*6,batch*6)).type_as(r_xi) #* torch.Size([30, 30])
        for i in range(batch):
            H_xi_xi[6*i:6*(i+1),6*i:6*(i+1)] = H_xi_xi_batch[i]
        def test_H():
            H_xi_xi_plot = H_xi_xi*1.0
            H_xi_xi_plot[H_xi_xi_plot<0]=0
            plot(H_xi_xi_plot,show=True)
        #* 右上角
        H_xi_rho_r_tmp = r_rho_r * r_xi #* torch.Size([5, 2, 45130, 6])
        H_xi_rho_r_batch = H_xi_rho_r_tmp.view(batch,-1,6).permute(1,0,2) #* torch.Size([90260, 5, 6])
        H_xi_rho_r = H_xi_rho_r_batch.reshape(-1,batch*6).permute(1,0) #* torch.Size([30, 90260]) 右上角
        # import pdb;pdb.set_trace()
        H_xi_rho_r = H_xi_rho_r.view(batch*6,-1,len_offset).sum(2)
        # import pdb;pdb.set_trace()
        #* 右下角
        H_rho_r_rho_r_tmp = r_rho_r**2 #* torch.Size([5, 2, 45130, 1])
        H_rho_r_rho_r = H_rho_r_rho_r_tmp.view(batch,-1).sum(0) #* torch.Size([90260])
        H_rho_r_rho_r = H_rho_r_rho_r.view(-1,len_offset).sum(1)
        # print(H_rho_r_rho_r.max())
        #! b 矩阵
        #* 位姿相关部分
        b_xi_tmp = r_xi * residual_edge[...,None] #* torch.Size([5, 2, 45130, 6])
        b_xi_tmp_batch = b_xi_tmp.view(batch,-1,6).permute(1,0,2) #* torch.Size([90260, 5, 6])
        b_xi = b_xi_tmp_batch.reshape(-1,batch*6).sum(0) #* torch.Size([30])
        #* 逆深度相关部分
        b_rho_r_tmp = r_rho_r * residual_edge[...,None] #* torch.Size([5, 2, 45130, 1])
        b_rho_r = b_rho_r_tmp.view(batch,-1).sum(0) #* torch.Size([90260])
        b_rho_r = b_rho_r.view(-1,len_offset).sum(1)
        #? 右下角的对角元素为0的位置对应的H矩阵的行列值为0，b对应位置也为0, 这样的话即使将对角元素值设为1，也应该不会影响最终的结果，但是矩阵就可逆了
        # (H_rho_r_rho_r<1e-3).sum()
        H_rho_r_rho_r_orig = H_rho_r_rho_r*1.0
        #* H_rho_r_rho_r_orig.max() tensor(8.5058e-07, device='cuda:0') 
        #* H_rho_r_rho_r_orig.min() 非零最小：tensor(4.4086e-19, device='cuda:0')
        cannot_update = H_rho_r_rho_r == 0
        H_rho_r_rho_r[cannot_update] = 1.0 #* 保证了0变为1不影响结果
        b_rho_r[cannot_update] = 0.0
        # b_rho_r[cannot_update].sum() #* 0
        # H_xi_rho_r[:,cannot_update].sum() #* 0
        #* b_rho_r.min() tensor(-0.0005, device='cuda:0')
        #* b_rho_r.max() tensor(0.0005, device='cuda:0')
        delta_rho_r = -(1.0/H_rho_r_rho_r) * b_rho_r #* torch.Size([45651])
        # import pdb;pdb.set_trace()
        delta_rho_r = delta_rho_r.view(1,-1)
        # print('H_rho_r_rho_r: ',H_rho_r_rho_r)
        # print('b_rho_r: ',b_rho_r)
        # print('delta_rho_r: ',delta_rho_r)
        updated_rho_r = rho_r + delta_rho_r
        updated_rho_r[updated_rho_r<1/40.0] = 1/40.0
        #* delta_rho_r.max() tensor(22606856., device='cuda:0') 
        #* delta_rho_r.min() tensor(-24823658., device='cuda:0')
        invalid_mask = abs(delta_rho_r)>0.5
        def plot_data():
            images_plot = []
            images_plot.append(['H_rho_r_rho_r_orig',to_plot_data(H_rho_r_rho_r_orig),'jet'])
            images_plot.append(['H_rho_r_rho_r',to_plot_data(H_rho_r_rho_r),'jet'])
            images_plot.append(['b_rho_r',to_plot_data(b_rho_r),'jet'])
            images_plot.append(['delta_rho_r',to_plot_data(delta_rho_r),'jet'])
            plt_imgs(images_plot,cols=1)
        # import pdb;pdb.set_trace()
        # print(len(H_rho_r_rho_r))
        # H_rho_r_rho_r_inv = torch.diag_embed(1.0/H_rho_r_rho_r) #* 一个矩阵居然占了7个G
        #* H * \delta x = - b 
        #* H = [ [H_xi_xi,      H_xi_rho_r],
        #*       [H_xi_rho_r^T, diag(H_rho_r_rho_r)] ]
        #* b = [b_xi,
        #*      b_rho_r]
        #* 边缘化你深度，求解位姿变化:
        #* (H_xi_xi - H_xi_rho_r * (H_rho_r_rho_r)^{-1} * H_rho_r_xi)\delat \xi = -(b_xi - H_xi_rho_r * (H_rho_r_rho_r)^{-1} * b_rho_r)
        #* H_rho_r_xi * \delat \xi + H_rho_r_rho_r * \delta \rho_r = -b_rho_r
        H_schur = H_xi_xi - (H_xi_rho_r * (1.0/H_rho_r_rho_r).unsqueeze(0) ) @ H_xi_rho_r.T
        b_schur = b_xi - (H_xi_rho_r * (1.0/H_rho_r_rho_r).unsqueeze(0) ) @ b_rho_r
        # delta_xi = torch.linalg.solve(H_xi_xi, - b_xi).reshape(batch,6)
        delta_xi = torch.linalg.solve(H_schur, - b_schur).reshape(batch,6)
        # torch.allclose(b_schur,H_schur @ delta_xi) #* True
        delta_T = xi2T(delta_xi) #* torch.Size([1, 4, 4])
        if iter_n>0:
            # delta_T[0,0,3] = 0.05 if iter_n==1 else  delta_T[0,0,3]
            T_c_w[1:] = torch.matmul(delta_T, T_c_w[1:])
            last_depth = depth*1.0
            # depth = 1.0/updated_rho_r
            # depth_samples[0][ref_edge_used_mask[None,...].repeat(ndepth,1,1)] = depth.view(-1)
        if iter_n%4==0 or iter_n==1:
            def plot_result():
                # ref_edge_used_mask = ref_edge>=ref_edge.min()
                yy, xx = torch.meshgrid([ torch.arange(0, height, dtype=dtype, device=device), torch.arange(0, width, dtype=dtype, device=device),])
                # yy[~ref_edge_used_mask] = 0.0
                # xx[~ref_edge_used_mask] = 0.0
                y, x = yy.contiguous().view(-1), xx.contiguous().view(-1)
                depth_all = depth_samples.squeeze(0)
                depth = depth_all.view(ndepth,-1) #* torch.Size([2, 45130])
                #* d*p = K*P_c
                uv1 = torch.stack((x, y, torch.ones_like(x))) #* torch.Size([3, 45130])
                uvd = uv1.unsqueeze(0) * depth.unsqueeze(1) #* torch.Size([2, 3, 45130])
                xyz_r = torch.matmul(K_inv,uvd) #* torch.Size([2, 3, 45130]) 参考相机坐标系下的坐标
                #* 投影到世界坐标系 P_w = R_w_c * P_c + t_w_c
                xyz_w = torch.matmul(T_w_c[0,:3,:3],xyz_r) + T_w_c[0,:3,3:] #* torch.Size([2, 3, 45130])
                xyz_w_batch = xyz_w.unsqueeze(0).repeat(batch,1,1,1) #* torch.Size([5, 2, 3, 45130]) [batch, ndepth, 3, N]
                #* 投影到多张源图相机坐标系
                xyz_s_batch = torch.matmul(T_c_w[1:,:3,:3].unsqueeze(1),xyz_w_batch) + T_c_w[1:,:3,3:].unsqueeze(1) #* torch.Size([5, 2, 3, 45130])
                #* 转为像素坐标
                uvd_s_batch = torch.matmul(K,xyz_s_batch)
                #* 错误深度mask
                negative_depth_mask = uvd_s_batch[:,:,2,:] <= 1e-3 #* torch.Size([5, 2, 45130])
                uvd_s_batch[:,:,0,:][negative_depth_mask] = float(width)
                uvd_s_batch[:,:,1,:][negative_depth_mask] = float(height)
                uvd_s_batch[:,:,2,:][negative_depth_mask] = 1.0
                grid_s_uv = uvd_s_batch[:,:,:2:,:] / uvd_s_batch[:,:,2:3,:] #* torch.Size([5, 2, 2, 45130])
                grid_s_uv_u_normalized = grid_s_uv[:,:,0,:] / ((width - 1) / 2) - 1
                grid_s_uv_v_normalized = grid_s_uv[:,:,1,:] / ((height - 1) / 2) - 1 #* torch.Size([5, 2, 45130])
                grid_s_uv_normalized = torch.stack((grid_s_uv_u_normalized, grid_s_uv_v_normalized), dim=3) #* torch.Size([5, 2, 45130, 2])
                src_edge_feats_sample = F.grid_sample(edge_feats[1:],grid_s_uv_normalized,mode="bilinear",padding_mode="zeros",align_corners=True,) #* torch.Size([5, 3, 2, 45130])
                ref_edge_plot = edge_feats[0,0]*1.0
                # ref_edge_plot[~ref_edge_used_mask] = 0.0
                plt_sec(ref_edge_plot[None,None],src_edge_feats_sample[0,0,0,:].view(1,1,height, width),show=True)
            # if True:
            def plot_depth_result(new_n_depth = 20,delta_d = 0.5):
                yy, xx = torch.meshgrid([ torch.arange(0, height, dtype=dtype, device=device), torch.arange(0, width, dtype=dtype, device=device),])
                y, x = yy[ref_edge_used_mask].contiguous().view(-1), xx[ref_edge_used_mask].contiguous().view(-1)
                ref_edge_used = ref_edge[ref_edge_used_mask] #* torch.Size([45130])
                # new_n_depth = 1
                # delta_d = 0
                d_samples = torch.linspace(1-delta_d,1+delta_d,new_n_depth, dtype=dtype, device=device)
                depth_all = depth_samples.squeeze(0).repeat(new_n_depth,1,1)
                if new_n_depth>1:
                    depth_all = depth_all * d_samples[...,None,None]
                depth = depth_all[ref_edge_used_mask[None,...].repeat(new_n_depth,1,1)].view(new_n_depth,-1) #* torch.Size([2, 45130])
                #* d*p = K*P_c
                uv1 = torch.stack((x, y, torch.ones_like(x))) #* torch.Size([3, 45130])
                uvd = uv1.unsqueeze(0) * depth.unsqueeze(1) #* torch.Size([2, 3, 45130])
                xyz_r = torch.matmul(K_inv,uvd) #* torch.Size([2, 3, 45130]) 参考相机坐标系下的坐标
                #* 投影到世界坐标系 P_w = R_w_c * P_c + t_w_c
                xyz_w = torch.matmul(T_w_c[0,:3,:3],xyz_r) + T_w_c[0,:3,3:] #* torch.Size([2, 3, 45130])
                xyz_w_batch = xyz_w.unsqueeze(0).repeat(1,1,1,1) #* torch.Size([5, 2, 3, 45130]) [batch, new_n_depth, 3, N]
                #* 投影到多张源图相机坐标系
                xyz_s_batch = torch.matmul(T_c_w[1:2,:3,:3].unsqueeze(1),xyz_w_batch) + T_c_w[1:2,:3,3:].unsqueeze(1) #* torch.Size([5, 2, 3, 45130])
                #* 转为像素坐标
                uvd_s_batch = torch.matmul(K,xyz_s_batch) #* torch.Size([1, 1, 3, 45652])
                #* 错误深度mask
                negative_depth_mask = uvd_s_batch[:,:,2,:] <= 1e-3 #* torch.Size([5, 2, 45130])
                uvd_s_batch[:,:,0,:][negative_depth_mask] = float(width)
                uvd_s_batch[:,:,1,:][negative_depth_mask] = float(height)
                uvd_s_batch[:,:,2,:][negative_depth_mask] = 1.0
                grid_s_uv = uvd_s_batch[:,:,:2:,:] / uvd_s_batch[:,:,2:3,:] #* torch.Size([1, 5, 2, 2])
                # import pdb;pdb.set_trace()
                grid_s_uv_squeeze = grid_s_uv.permute(0,2,1,3).reshape(2,-1).squeeze().cpu().long().numpy() #* torch.Size([2, 45652])
                grid_s_uv_squeeze_valid_mask = (grid_s_uv_squeeze[0]>0)*(grid_s_uv_squeeze[0]<width-1)*(grid_s_uv_squeeze[1]>0)*(grid_s_uv_squeeze[1]<height-1)
                src_edge = edge_feats[1,0].cpu().numpy()
                zeors = np.zeros_like(src_edge)
                src_edge_g = np.stack((zeors,src_edge,zeors),axis=2)
                x_p = grid_s_uv_squeeze[0][grid_s_uv_squeeze_valid_mask]
                y_p = grid_s_uv_squeeze[1][grid_s_uv_squeeze_valid_mask]
                src_edge_g[y_p,x_p] = [1,0,0]
                ref_edge_p = (ref_edge*1.0).squeeze().cpu().numpy()
                ref_edge_g = np.stack((zeors,ref_edge_p,zeors),axis=2) #* (1024, 1280, 3)
                ref_edge_g[ref_edge_used_mask.cpu().numpy(),:]=[1,0,0]
                images_plot = []
                images_plot.append(['ref_edge_g',to_plot_data(ref_edge_g),'jet'])
                images_plot.append(['src_edge_g',to_plot_data(src_edge_g),'jet'])
                if len(depth_on_src)==0:
                    depth_on_src.append(['src_edge_g_'+str(iter_n),to_plot_data(src_edge_g),'jet'])
                if len(depth_on_src)>0 and 'src_edge_g_'+str(iter_n) not in depth_on_src[-1][0]:
                    depth_on_src.append(['src_edge_g_'+str(iter_n),to_plot_data(src_edge_g),'jet'])
                plt_imgs(images_plot,sharexy=False,show=False)
                plt_imgs(depth_on_src,show=True)
                # import pdb;pdb.set_trace()
            plot_depth_result(new_n_depth = 1,delta_d = 0)
            # plot_result()
            import pdb;pdb.set_trace()
            # def plot_src_edge_feats(select_ths=0.9): #* 查看投影结果
            #     images_plot = []
            #     ref_edge_test = edge_feats[0,0]*1.0
            #     ref_edge_used_mask = ref_edge_test>select_ths
            #     can_update = H_rho_r_rho_r>0
            #     cannot_update = H_rho_r_rho_r==0
            #     images_plot.append(['ref_edge_test',to_plot_data(ref_edge_test),'jet'])
            #     images_plot.append(['ref_edge_used_mask',to_plot_data(ref_edge_used_mask),'jet'])
            #     ref_edge_test[~ref_edge_used_mask] = 0
            #     ref_edge_test1= ref_edge_test*1.0
            #     ref_edge_test1[ref_edge_used_mask] = can_update.float()
            #     images_plot.append(['ref_edge_test1',to_plot_data(ref_edge_test1),'jet'])
            #     if debug:
            #         for i in range(src_edge_feats_sample.shape[0]):
            #             src_edge_feat2 = src_edge_feats_sample[i,0,0,:].view(height, width)
            #             images_plot.append(['src_edge_feat'+str(i),to_plot_data(src_edge_feat2),'jet'])
    import pdb;pdb.set_trace()
    pass

def xi2T(xi):
    if xi.shape[1]!=6:
        print("Shape should be nx6 !")
        return None
    xi_phi = xi[:,3:]
    xi_rho = xi[:,:3]
    xi_theta = torch.norm(xi_phi,dim=1, keepdim=True) #* torch.Size([1, 1])
    xi_a = xi_phi / xi_theta #* 单位向量
    T_list =[]
    for i in range(xi.shape[0]):
        # import pdb;pdb.set_trace()
        theta = xi_theta[i]
        a = xi_a[i]
        rho = xi_rho[i]
        R = (1-torch.cos(theta)) * a.reshape(-1,1) * a.reshape(1,-1) + torch.sin(theta)*skew(a) + torch.cos(theta) * torch.eye(3).type_as(xi)
        J = torch.sin(theta)/theta * torch.eye(3).type_as(xi) + (1-torch.cos(theta))/theta * skew(a) + (1-torch.sin(theta)/theta)*a.reshape(-1,1) * a.reshape(1,-1)
        t = J @ rho
        T = torch.eye(4).type_as(xi)
        T[:3,:3] = R
        T[:3,3] = t
        T_list.append(T)
    return torch.stack(T_list,dim=0)

def skew(phi):
    if len(phi)!=3:
        print("length should be 3")
        return None
    return torch.tensor([[0,-phi[2],phi[1]],[phi[2],0,-phi[0]],[-phi[1],phi[0],0]]).type_as(phi)

def cal_wider_gray_grad(ref_image, offset = None, win_size = 3,ths=0.1, step=1):
    batch,channels,height,width = ref_image.shape
    if channels == 3 :
        ref_image = ref_image.half()
        gray_image = ref_image[:,0:1]*0.299 + ref_image[:,1:2]*0.587 + ref_image[:,2:3]*0.114
    elif channels == 1:
        gray_image = ref_image.half()
    else:
        raise NotImplementedError
    dtype = gray_image.dtype
    device = gray_image.device
    gray_image = gray_image*1.0
    if offset is None:
        win_offset = []
        for i in range(0,win_size,step):
            for j in range(0,win_size,step):
                win_offset.append([j- win_size//2,i- win_size//2])
        win_offset.sort(key=lambda p : p[0]**2+p[1]**2)
        all_offset_torch = torch.tensor(win_offset,dtype=dtype, device=device)
    else:
        offset.sort(key=lambda p : p[0]**2+p[1]**2)
        all_offset_torch = torch.tensor(offset,dtype=dtype, device=device) #* torch.Size([9, 2])
    # import pdb;pdb.set_trace()
    all_offset_map_torch = all_offset_torch[None,None,...].repeat(batch,4,1,1).permute(0,3,1,2) #* torch.Size([batch, 2, 4, 9])
    new_h, new_w = all_offset_map_torch.shape[-2:]
    # offset_grid = get_offset_grid(length=1,only4=True)
    # offset_grid_torch_list = [torch.tensor(offset, dtype=dtype, device=device) for offset in offset_grid]
    # #* 用于采样，返回坐标
    # all_offset_map_torch = torch.stack(offset_grid_torch_list,dim=0) #* torch.Size([4, 56, 2])
    #* 开始构建图像坐标
    y_grid, x_grid = torch.meshgrid([torch.arange(0, height, dtype=dtype, device=device), torch.arange(0, width, dtype=dtype, device=device),])
    y_grid, x_grid = y_grid.contiguous().view(height * width), x_grid.contiguous().view(height * width)
    xy = torch.stack((x_grid, y_grid))  # [2, H*W]
    xy = torch.unsqueeze(xy, 0).repeat(batch, 1, 1)  # [B, 2, H*W]
    xy_list = []
    #* 为每个点添加偏置坐标
    for i in range(len(all_offset_torch)):
        xy_list.append((xy + all_offset_torch[i].view(2,1)).unsqueeze(2))
    xy_with_offset = torch.cat(xy_list, dim=2)  # [B, 2, 9, H*W]
    x_normalized = xy_with_offset[:, 0, :, :] / ((width - 1) / 2) - 1
    y_normalized = xy_with_offset[:, 1, :, :] / ((height - 1) / 2) - 1
    grid = torch.stack((x_normalized, y_normalized), dim=3)  # [B, 9, H*W, 2]
    #* 归一化坐标采样
    surrounding_grays = F.grid_sample(gray_image, grid, mode="nearest", padding_mode="reflection", align_corners=True).view(batch, len(all_offset_torch), height, width)
    #* surrounding_grays.shape torch.Size([6, 9, 1024, 1280])
    
    surrounding_grays_mask = surrounding_grays>ths
    surrounding_grays_mask_max=torch.max(surrounding_grays_mask,dim=1)
    surrounding_grays_mask_max_indices = surrounding_grays_mask_max.indices.half() #* torch.Size([6, 1024, 1280])
    zeros = torch.zeros_like(surrounding_grays_mask_max_indices)
    surrounding_grays_mask_max_indices_uv = torch.stack([zeros,surrounding_grays_mask_max_indices],dim=3) #* torch.Size([6, 1024, 1280, 2])
    uv_x_normalized = surrounding_grays_mask_max_indices_uv[:,:,:,1] / ((new_w - 1) / 2) - 1
    uv_y_normalized = surrounding_grays_mask_max_indices_uv[:,:,:,0] / ((new_h - 1) / 2) - 1
    surrounding_grays_mask_max_indices_grid = torch.stack((uv_x_normalized, uv_y_normalized), dim=3)
    #* 得到每个像素的周围最近且大于ths的点在all_offset_map_torch中的坐标
    surrounding_grays_max_uv = F.grid_sample(all_offset_map_torch, surrounding_grays_mask_max_indices_grid, mode="nearest", padding_mode="reflection", align_corners=True)
    #* surrounding_grays_max_uv.shape torch.Size([6, 2, 1024, 1280])
    # gray_image[0,:,492-3:492+4,653-3:653+4]
    def plot_arrow(u=1068,v=278,win_size=5,step=2):
        plot(gray_image[0])
        for i in range(u-win_size,u+win_size+1,step):
            for j in range(v-win_size,v+win_size+1,step):
                plt.arrow(i,j,surrounding_grays_max_uv[0,0,j,i].cpu().numpy(),surrounding_grays_max_uv[0,1,j,i].cpu().numpy(),width=0.2)
                print(i,j,surrounding_grays_max_uv[0,0,j,i].cpu().numpy(),surrounding_grays_max_uv[0,1,j,i].cpu().numpy())
        plt.show()
    # surrounding_grays_max_uv[0,:,492,653]
    #* 每个点在原图中的坐标，重新采样得到最近且大于ths的那个值
    surrounding_grays_max_value_uv = surrounding_grays_max_uv + xy.view(batch,2,height, width)
    value_x_normalized = surrounding_grays_max_value_uv[:,0,:,:] / ((width - 1) / 2) - 1
    value_y_normalized = surrounding_grays_max_value_uv[:,1,:,:] / ((height -1) / 2) - 1
    value_uv_grid = torch.stack((value_x_normalized, value_y_normalized), dim=3) #* torch.Size([6, 1024, 1280, 2])
    #* 得到距离每个点的最近且大于ths的那个值
    surrounding_grays_max_value = F.grid_sample(gray_image, value_uv_grid, mode="nearest", padding_mode="reflection", align_corners=True).view(batch, -1, height, width)
    #* 有了值和相应的坐标，就可计算梯度了
    diff_value = surrounding_grays_max_value - gray_image #* 灰度差异 torch.Size([6, 1, 1024, 1280])
    delta_u = surrounding_grays_max_uv[:,0,:,:]
    delta_v = surrounding_grays_max_uv[:,1,:,:]
    delta_uv = delta_u**2+delta_v**2 #* torch.Size([6, 1024, 1280])
    delta_uv_mask = delta_uv==0
    grad_uv = diff_value.squeeze()/(1e-5+delta_uv)
    grad_uv[delta_uv_mask] = 0
    grad_u = grad_uv*delta_u
    grad_v = grad_uv*delta_v
    gray_grad_xy = torch.stack((grad_u,grad_v), dim=1) #* torch.Size([6, 2, 1024, 1280])
    squared_gray_grad = gray_grad_xy[:,0:1]*gray_grad_xy[:,0:1] + gray_grad_xy[:,1:]*gray_grad_xy[:,1:]
    # import pdb;pdb.set_trace()
    return gray_image, squared_gray_grad, gray_grad_xy
    import pdb;pdb.set_trace()
    surrounding_squared_gray_grad = patch_grid_sample(squared_gray_grad,win_size=3) #* torch.Size([1, 1, 9, 1310720])
    surrounding_squared_gray_grad = surrounding_squared_gray_grad.view(batch,-1, height, width)
    squared_gray_grad_smoothed = surrounding_squared_gray_grad.sum(dim=1)/9.0
    return gray_image, squared_gray_grad_smoothed.unsqueeze(1), gray_grad_xy

def plt_imgs(images_plot,show=True,save=False,cols=2,sharexy=True):
    num_image=len(images_plot)
    rows=num_image//cols+1 if num_image%cols>0 else num_image//cols
    sharey = True if cols>1 else False
    if sharexy:
        fig, ax = plt.subplots(rows, cols, figsize=(rows*6, cols*5), sharex=True, sharey=sharey)
    else:
        fig, ax = plt.subplots(rows, cols, figsize=(rows*6, cols*5))
    ax=ax.reshape(-1)
    for row in range(rows):
        for col in range(cols):
            image_idx=row*cols+col
            if image_idx>=num_image:
                ax[row*cols+col].set_axis_off()
            else:
                name,image,cmap=images_plot[image_idx]
                # import pdb;pdb.set_trace()
                if len(image.shape)>1:
                    ax[image_idx].imshow(image,cmap=cmap)
                    ax[image_idx].set_title(name)
                    ax[image_idx].set_axis_off()
                else:
                    ax[image_idx].plot(image)
                    ax[image_idx].set_title(name)
                    # ax[image_idx].set_axis_off()
    plt.tight_layout()
    if save:
        if not os.path.exists('tmp'):
            os.makedirs("tmp")
        plt.savefig('tmp/images_plot.png',dpi=500) #! figsize=(10,10) resolution=(500*10,500*10)=(5k,5k)
        print("Saving to tmp/images_plot.png")
    if show:
        plt.show()

def cal_gray_grad(ref_image):
    batch,channels,height,width = ref_image.shape
    if channels == 3 :
        ref_image = ref_image.half()
        gray_image = ref_image[:,0:1]*0.299 + ref_image[:,1:2]*0.587 + ref_image[:,2:3]*0.114
    elif channels == 1:
        gray_image = ref_image.half()
    else:
        raise NotImplementedError
    dtype = gray_image.dtype
    device = gray_image.device
    gray_image = gray_image*1.0
    offset_grid = get_offset_grid(length=1,only4=True)
    offset_grid_torch_list = [torch.tensor(offset, dtype=dtype, device=device) for offset in offset_grid]
    #* 用于采样，返回坐标
    all_offset_map_torch = torch.stack(offset_grid_torch_list,dim=0) #* torch.Size([4, 56, 2])
    #* 用于添加偏置
    all_offset_torch = all_offset_map_torch.view(-1,2)
    #* 开始构建图像坐标
    y_grid, x_grid = torch.meshgrid([torch.arange(0, height, dtype=dtype, device=device), torch.arange(0, width, dtype=dtype, device=device),])
    y_grid, x_grid = y_grid.contiguous().view(height * width), x_grid.contiguous().view(height * width)
    xy = torch.stack((x_grid, y_grid))  # [2, H*W]
    xy = torch.unsqueeze(xy, 0).repeat(batch, 1, 1)  # [B, 2, H*W]
    xy_list = []
    #* 为每个点添加偏置坐标
    for i in range(len(all_offset_torch)):
        xy_list.append((xy + all_offset_torch[i].view(2,1)).unsqueeze(2))
    xy_with_offset = torch.cat(xy_list, dim=2)  # [B, 2, 9, H*W]
    x_normalized = xy_with_offset[:, 0, :, :] / ((width - 1) / 2) - 1
    y_normalized = xy_with_offset[:, 1, :, :] / ((height - 1) / 2) - 1
    grid = torch.stack((x_normalized, y_normalized), dim=3)  # [B, 9, H*W, 2]
    #* 归一化坐标采样
    #* ref_image [1,3,1024,1280]
    #* surrounding_colors: torch.Size([1, 3, 4, 1, 1024, 1280])
    # surrounding_colors = F.grid_sample(ref_image, grid, mode="nearest", padding_mode="reflection", align_corners=True).view(batch, -1, 4,len(all_offset_torch)//4, height, width)
    #* surrounding_grays.shape torch.Size([1, 1, 4, 1, 1024, 1280])
    surrounding_grays = F.grid_sample(gray_image, grid, mode="nearest", padding_mode="reflection", align_corners=True).view(batch, -1, 4,len(all_offset_torch)//4, height, width)
    # color_grad_xy = surrounding_colors[:,:,0:2] - surrounding_colors[:,:,2:4] #* torch.Size([1, 3, 2, 1, 1024, 1280])
    gray_grad_xy = surrounding_grays[:,:,0:2] - surrounding_grays[:,:,2:4] 	  #* torch.Size([1, 1, 2, 1, 1024, 1280])
    # color_grad_xy = color_grad_xy.squeeze(3)*0.5 #* torch.Size([1, 3, 2, 1024, 1280])
    gray_grad_xy = gray_grad_xy.squeeze(3)*0.5   #* torch.Size([1, 1, 2, 1024, 1280])
    squared_gray_grad = gray_grad_xy[:,0,0]*gray_grad_xy[:,0,0] + gray_grad_xy[:,0,1]*gray_grad_xy[:,0,1] #* torch.Size([1, 1024, 1280])
    squared_gray_grad = squared_gray_grad.unsqueeze(1) #* torch.Size([1, 1, 1024, 1280])
    surrounding_squared_gray_grad = patch_grid_sample(squared_gray_grad,win_size=3) #* torch.Size([1, 1, 9, 1310720])
    surrounding_squared_gray_grad = surrounding_squared_gray_grad.view(batch,-1, height, width)
    squared_gray_grad_smoothed = surrounding_squared_gray_grad.sum(dim=1)/9.0
    return gray_image, squared_gray_grad_smoothed.unsqueeze(1), gray_grad_xy.squeeze(1)

def cal_feature_img(ref_image):
    batch,channels,height,width = ref_image.shape
    feature_offset = [[0,0]]
    for i in range(1,4):
        feature_offset.append([0,i])
        feature_offset.append([0,-i])
        feature_offset.append([i,0])
        feature_offset.append([-i,0])
    for i in range(1,3):
        feature_offset.append([i,i])
        feature_offset.append([-i,-i])
        feature_offset.append([i,-i])
        feature_offset.append([-i,i])
    # plt_select_mask(feature_offset,length=3)
    # feature_gray = patch_grid_sample(gray_image,offset=feature_offset) #* torch.Size([1, 1, 21, 1310720])
    # feature_gray = feature_gray.view(batch,-1, height, width) #* torch.Size([1, 21, 1024, 1280])
    feature_color = patch_grid_sample(ref_image,offset=feature_offset) #* torch.Size([1, 3, 21, 1310720])
    feature_color = feature_color.view(batch,-1, height, width) #* torch.Size([1, 63, 1024, 1280])
    feature_offset_np = np.array(feature_offset)
    return feature_color, feature_offset_np

def plt_pattern(ref_image,feature_color,feature_offset_np,uvs=[[160,582]],feature_color_src=None):
    if ref_image.shape[1]==1:
        plot(ref_image,cmap='gray')
    else:
        plot(ref_image)
    num_pattern = 16
    radius = 10
    thetas = np.linspace(0,2*np.pi,num_pattern,endpoint=False)
    pattern_offset = [[np.cos(theta), np.sin(theta)] for theta in thetas] + [[0,0]]
    pattern_offset_np = np.array(pattern_offset)*radius
    # plt.axis([u-16,u+16, v-16,v+16])
    for uv in uvs:
        u,v =uv
        ref_feat = feature_color[0,:,v,u]
        for po in pattern_offset_np:
            if feature_color_src is not None:
                src_feat = feature_color_src[0,:,int(v+po[1]),int(u+po[0])]
            else:
                src_feat = feature_color[0,:,int(v+po[1]),int(u+po[0])]
            l2norm = (ref_feat-src_feat).norm()/ref_feat.norm()
            costheta = (ref_feat*src_feat).sum()/(ref_feat.norm()*src_feat.norm())
            simil = l2norm.cpu().numpy()
            # import pdb;pdb.set_trace()
            plt.scatter(feature_offset_np[:,0]+u+po[0],feature_offset_np[:,1]+v+po[1])
            plt.plot([u,u+po[0]],[v,v+po[1]])
            plt.text(u+po[0], v+po[1], str(simil),c='w')

def plt_sec(ref_image, src_features, length=40,show=False):
    batch,channels,height,width = ref_image.shape
    if channels==3:
        src_features_c = src_features[:,[2,1,0],:,:].half()
    else:
        src_features_c = src_features.half()
    src_features_c_s = src_features_c*1.0
    ref_image_s = ref_image*1.0
    for i in range(0,height-length,length):
        src_features_c_s[:,:,i:i+length//2,:]=0
        ref_image_s[:,:,i+length//2:i+length,:]=0
    # plot(src_features_c_s)
    add_image = (src_features_c_s+ref_image_s)*2.0
    add_image_hori = add_image/add_image.max()
    # plot(add_image_hori)
    # import pdb;pdb.set_trace()
    src_features_c_s = src_features_c*1.0
    ref_image_s = ref_image*1.0
    for i in range(0,width-length,length):
        src_features_c_s[:,:,:,i:i+length//2]=0
        ref_image_s[:,:,:,i+length//2:i+length]=0
    add_image = (src_features_c_s+ref_image_s)*2.0
    add_image_vert = add_image/add_image.max()
    # plot(add_image_vert)
    if show:
        plot(add_image_hori)
        plot(add_image_vert)
        plt.show()
    return add_image_hori, add_image_vert

def proj_ref2src(ref_image_thsed,src_uv):
    batch,channels,height,width = ref_image_thsed.shape
    #* src_uv.shape  torch.Size([1, 2, 1, 1024, 1280])  (batch, 2, num_depth, height, width)
    src_uv = src_uv.squeeze() #* torch.Size([2, 1024, 1280])
    src_uv_get_neighbors = src_uv.unsqueeze(1).repeat(1,4,1,1) #* torch.Size([2, 4, 1024, 1280])
    src_uv_get_neighbors[:,0] = torch.floor(src_uv)
    src_uv_get_neighbors[0,1] = torch.floor(src_uv[0])
    src_uv_get_neighbors[1,1] = torch.ceil(src_uv[1])
    src_uv_get_neighbors[0,2] = torch.ceil(src_uv[0])
    src_uv_get_neighbors[1,2] = torch.floor(src_uv[1])
    src_uv_get_neighbors[:,3] = torch.ceil(src_uv)
    src_uv_get_neighbors = src_uv_get_neighbors.long()
    # ref_image_neighbors = ref_image.squeeze().unsqueeze(1).repeat(1,4,1,1) #* torch.Size([3, 4, 1024, 1280])
    ref_image_neighbors = ref_image_thsed.view(channels,height,width).unsqueeze(1).repeat(1,4,1,1) #* torch.Size([3, 4, 1024, 1280])
    src_uv_get_neighbors_valid_mask = (src_uv_get_neighbors[0]>=0) * (src_uv_get_neighbors[0]<width) * (src_uv_get_neighbors[1]>=0) * (src_uv_get_neighbors[1]<height)
    rel_ref_image_neighbors = ref_image_neighbors[:,src_uv_get_neighbors_valid_mask] #* torch.Size([3, 1595144])
    src_uv_get_neighbors_valid = src_uv_get_neighbors[:,src_uv_get_neighbors_valid_mask] #* torch.Size([2, 1595144])
    src_uv_get_neighbors_valid_flatten = src_uv_get_neighbors_valid[1]*width + src_uv_get_neighbors_valid[0]
    src_uv_get_neighbors_valid_flatten_unique, src_uv_get_neighbors_valid_flatten_pinds, src_uv_get_neighbors_valid_flatten_pcounts = torch.unique(src_uv_get_neighbors_valid_flatten, return_inverse=True, return_counts=True)
    #* pinds 为原始数据在unique_flat_ids中的索引 torch.Size([398768])
    #* pcounts 不同元素的个数 torch.Size([350920])
    rel_ref_image_neighbors_mean = scatter_mean(rel_ref_image_neighbors, src_uv_get_neighbors_valid_flatten_pinds.unsqueeze(0)) #* torch.Size([3, 826013])
    #* 根据index，将index相同值对应的src元素进行对应定义的计算，dim为在第几维进行相应的运算。e.g.scatter_sum即进行sum运算，scatter_mean即进行mean运算。
    proj_image = torch.zeros_like(ref_image_thsed)
    proj_image_flatten = proj_image.view(batch,-1,height*width)
    proj_image_flatten[0,:,src_uv_get_neighbors_valid_flatten_unique.long()] = rel_ref_image_neighbors_mean
    proj_image = proj_image_flatten.view(batch,-1,height, width)
    return proj_image

def key_info_filter(ref_image,squared_gray_grad,ths_percent = 0.9):
    batch,channels,height,width = ref_image.shape
    # ths_percent = 0.9
    global_ths = torch.kthvalue(squared_gray_grad.view(batch),int(height*width*ths_percent)) #* global_ths.values.shape torch.Size([1, 1])
    squared_gray_grad_thsed = squared_gray_grad*1.0
    squared_gray_grad_thsed[squared_gray_grad_thsed<global_ths.values] = 0
    # squared_gray_grad_thsed[squared_gray_grad_thsed>=global_ths.values[0,0]] = 1
    # plot(squared_gray_grad_thsed)
    surrounding_squared_gray_grad_thsed = patch_grid_sample(squared_gray_grad_thsed,win_size=3) #* torch.Size([1, 1, 9, 1310720])
    surrounding_squared_gray_grad_thsed = surrounding_squared_gray_grad_thsed.view(batch,-1, height, width)
    squared_gray_grad_thsed_thsed = surrounding_squared_gray_grad_thsed.sum(dim=1)
    global_ths = torch.kthvalue(squared_gray_grad_thsed_thsed.view(-1),int(height*width*ths_percent))
    squared_gray_grad_thsed_thsed_plot = squared_gray_grad_thsed_thsed*1.0
    squared_gray_grad_thsed_thsed_plot[squared_gray_grad_thsed_thsed_plot<global_ths.values]=0
    squared_gray_grad_thsed_thsed_plot[squared_gray_grad_thsed_thsed_plot>=global_ths.values]=1
    ref_image_thsed = ref_image * squared_gray_grad_thsed_thsed_plot.unsqueeze(1)
    plot(squared_gray_grad_thsed_thsed_plot)
    plt.show()
    return ref_image_thsed

def dso_pts_select(ref_image,squared_gray_grad,ths_percent = 0.9, use_median=False,enhance_win_size=3):
    #* 首先将图像划分为32*32的区间
    #* 下面采用网格中位数来选择，感觉不太好，暂时不用
    #* 阈值法则采用每个区间的梯度最大10%作为关键信息，效果还可以
    dtype, device = ref_image.dtype, ref_image.device
    batch,channels,height,width = ref_image.shape
    win_size = 32
    win_y_grid, win_x_grid = torch.meshgrid([torch.arange(win_size//2, height, win_size, dtype=dtype, device=device), torch.arange(win_size//2, width, win_size, dtype=dtype, 
    device=device),])
    new_h, new_w = win_y_grid.shape[0], win_y_grid.shape[1]
    win_y_grid, win_x_grid = win_y_grid.contiguous().view(-1), win_x_grid.contiguous().view(-1)
    win_xy = torch.stack((win_x_grid, win_y_grid))  # [2, H*W]
    win_xy = torch.unsqueeze(win_xy, 0).repeat(batch, 1, 1)  # [B, 2, H*W]
    win_xy_list = []
    win_offset = []
    for i in range(win_size):
        for j in range(win_size):
            win_offset.append([j,i])
    win_offset_torch = torch.tensor(win_offset,dtype=dtype, device=device)- win_size//2 #* torch.Size([1024, 2])
    # import pdb;pdb.set_trace()
    #* 为每个点添加偏置坐标
    for i in range(len(win_offset_torch)):
        win_xy_list.append((win_xy + win_offset_torch[i].view(2,1)).unsqueeze(2))
    win_xy_with_offset = torch.cat(win_xy_list, dim=2)  #* torch.Size([1, 2, 1024, 1280])
    win_x_normalized = win_xy_with_offset[:, 0, :, :] / ((width - 1) / 2) - 1
    win_y_normalized = win_xy_with_offset[:, 1, :, :] / ((height - 1) / 2) - 1
    win_grid = torch.stack((win_x_normalized, win_y_normalized), dim=3)  #* torch.Size([1, 1024, 1280, 2])
    # import pdb;pdb.set_trace()
    #* squared_gray_grad_patch.shape torch.Size([1, 1, 1024, 1280]) #* 总共1280个区域，每个区域共有1024个点，每个点一个平方灰度梯度
    squared_gray_grad_patch = F.grid_sample(squared_gray_grad, win_grid, mode="nearest", padding_mode="reflection", align_corners=True)
    def plot_grad_line(index):
        plt.figure()
        test_patch  = squared_gray_grad_patch[0,0,:,index].cpu().float().numpy()
        plt.plot(test_patch)
        plt.plot([0,1000],[48,48])
    if use_median:
        squared_gray_grad_patch_median = squared_gray_grad_patch.median(2) #* squared_gray_grad_patch_median[0].shape torch.Size([1, 1, 1280])
        #* 梯度中位数图
        squared_gray_grad_patch_median = squared_gray_grad_patch_median[0].view(batch, -1, new_h, new_w) #* torch.Size([1, 1, 32, 40])
        squared_gray_grad_patch_median_ths = 48 #* 48*48 = 2304
        squared_gray_grad_patch_median[squared_gray_grad_patch_median>squared_gray_grad_patch_median_ths] = squared_gray_grad_patch_median_ths
    else:
        num_pts = squared_gray_grad_patch.shape[2]
        squared_gray_grad_patch_ths_kth = torch.kthvalue(squared_gray_grad_patch,int(num_pts*ths_percent),2)
        #* squared_gray_grad_patch_ths.values.shape torch.Size([1, 1, 1280])
        # import pdb;pdb.set_trace()
        squared_gray_grad_patch_ths = squared_gray_grad_patch_ths_kth.values.view(batch, -1, new_h, new_w)
    # plt.plot(squared_gray_grad.view(-1).sort().values.cpu().float().numpy())
    # plt.plot([0,1280*1024],[48*48,48*48])
    # plot(ref_image)
    # # plot(gray_image,cmap='gray')
    # u, v = 16, 16
    # plot(ref_image[0,:,v-16:v+16,u-16:u+16])
    # plot(gray_image[0,:,v-16:v+16,u-16:u+16],cmap='gray')
    # plot(squared_gray_grad[0,:,v-16:v+16,u-16:u+16])
    # plot_grad_line(0)
    # plot_grad_line(1)
    # plt.show()
    smooth_y_grid, smooth_x_grid = torch.meshgrid([torch.arange(0, new_h, dtype=dtype, device=device), torch.arange(0, new_w, dtype=dtype, device=device),])
    smooth_y_grid, smooth_x_grid = smooth_y_grid.contiguous().view(-1), smooth_x_grid.contiguous().view(-1)
    smooth_xy = torch.stack((smooth_x_grid, smooth_y_grid))  # [2, H*W]
    smooth_xy = torch.unsqueeze(smooth_xy, 0).repeat(batch, 1, 1)  # [B, 2, H*W]
    smooth_xy_list = []
    smooth_offset = []
    smooth_size = 3
    for i in range(smooth_size):
        for j in range(smooth_size):
            smooth_offset.append([j,i])
    smooth_offset_torch = torch.tensor(smooth_offset,dtype=dtype, device=device)- smooth_size//2 #* torch.Size([1024, 2])
    # import pdb;pdb.set_trace()
    #* 为每个点添加偏置坐标
    for i in range(len(smooth_offset_torch)):
        smooth_xy_list.append((smooth_xy + smooth_offset_torch[i].view(2,1)).unsqueeze(2))
    smooth_xy_with_offset = torch.cat(smooth_xy_list, dim=2)  #* torch.Size([1, 2, 1024, 1280])
    smooth_x_normalized = smooth_xy_with_offset[:, 0, :, :] / ((new_w - 1) / 2) - 1
    smooth_y_normalized = smooth_xy_with_offset[:, 1, :, :] / ((new_h - 1) / 2) - 1
    smooth_grid = torch.stack((smooth_x_normalized, smooth_y_normalized), dim=3)  #* torch.Size([1, 9, 1280, 2])
    # import pdb;pdb.set_trace()
    #* squared_gray_grad_patch.shape torch.Size([1, 1, 1024, 1280]) #* 总共1280个区域，每个区域共有1024个点，每个点一个平方灰度梯度
    if use_median:
        squared_gray_grad_patch_median_smooth = F.grid_sample(squared_gray_grad_patch_median, smooth_grid, mode="nearest", padding_mode="reflection", align_corners=True)
        #* 用每个网格周围的8个点来平滑滤波
        squared_gray_grad_patch_median_smooth = squared_gray_grad_patch_median_smooth.view(batch, -1, new_h, new_w) #* torch.Size([1, 9, 32, 40])
        squared_gray_grad_patch_median_smoothed = squared_gray_grad_patch_median_smooth.mean(1).unsqueeze(1) #* torch.Size([1, 1, 32, 40])
        threshold_smmothed = squared_gray_grad_patch_median_smoothed * squared_gray_grad_patch_median_smoothed
    else:
        threshold_smmothed = squared_gray_grad_patch_ths
    ths_y_grid, ths_x_grid = torch.meshgrid([torch.arange(0, height, dtype=dtype, device=device), torch.arange(0, width, dtype=dtype, 
    device=device),])
    ths_y_grid, ths_x_grid = ths_y_grid.contiguous().view(-1), ths_x_grid.contiguous().view(-1)
    ths_xy = torch.stack((ths_x_grid, ths_y_grid))  # [2, H*W]
    ths_xy = torch.unsqueeze(ths_xy, 0).repeat(batch, 1, 1)  # [B, 2, H*W]
    ths_xy_with_offset = ths_xy.unsqueeze(2)  #* torch.Size([1, 2, 1024, 1280])
    ths_x_normalized = ths_xy_with_offset[:, 0, :, :] / ((width - 1) / 2) - 1
    ths_y_normalized = ths_xy_with_offset[:, 1, :, :] / ((height - 1) / 2) - 1
    ths_grid = torch.stack((ths_x_normalized, ths_y_normalized), dim=3)  #* torch.Size([1, 1, 1310720, 2])
    ths_for_each_pt = F.grid_sample(threshold_smmothed, ths_grid, mode="nearest", padding_mode="reflection", align_corners=True)
    ths_for_each_pt = ths_for_each_pt.view(batch,-1,height,width) #* torch.Size([1, 1, 1024, 1280])
    selected_mask = squared_gray_grad > ths_for_each_pt  #* torch.Size([1, 1, 1024, 1280])
    # return ref_image*selected_mask
    # plot(ref_image*selected_mask)
    #* 再对mask进行一次全局滤波，主要去除一些离散点
    # surrounding_selected_mask = patch_grid_sample(selected_mask.half(),win_size = enhance_win_size) #* torch.Size([1, 1, 9, 1310720])
    surrounding_selected_mask = patch_grid_sample(selected_mask.half()*squared_gray_grad,win_size = enhance_win_size) #* torch.Size([1, 1, 9, 1310720])
    surrounding_selected_mask = surrounding_selected_mask.view(batch,-1, height, width)
    surrounding_selected_mask_sum = surrounding_selected_mask.sum(dim=1)
    global_ths = torch.kthvalue(squared_gray_grad.view(-1),int(height*width*ths_percent))
    surrounding_selected_mask_sum_thsed = surrounding_selected_mask_sum*1.0
    surrounding_selected_mask_sum_thsed[surrounding_selected_mask_sum_thsed<global_ths.values]=0
    surrounding_selected_mask_sum_thsed[surrounding_selected_mask_sum_thsed>=global_ths.values]=1
    ref_image_thsed = ref_image * surrounding_selected_mask_sum_thsed.unsqueeze(1)
    # plot(surrounding_selected_mask_sum_thsed)
    # plot(ref_image_thsed)
    # plot(squared_gray_grad)
    # plt.show()
    # import pdb;pdb.set_trace()
    return ref_image_thsed

def patch_grid_sample(source, offset = None, win_size = 3, start=[0,0], end=[],step = 1):
    #* source [B,C,H,W]
    dtype, device = source.dtype, source.device
    bs, ch, height, width = source.shape
    win_y_grid, win_x_grid = torch.meshgrid([torch.arange(start[0], height, step, dtype=dtype, device=device), torch.arange(start[1], width, step, dtype=dtype, 
    device=device),])
    win_y_grid, win_x_grid = win_y_grid.contiguous().view(-1), win_x_grid.contiguous().view(-1)
    win_xy = torch.stack((win_x_grid, win_y_grid))  # [2, H*W]
    win_xy = torch.unsqueeze(win_xy, 0).repeat(bs, 1, 1)  # [B, 2, H*W]
    win_xy_list = []
    if offset is None:
        win_offset = []
        for i in range(win_size):
            for j in range(win_size):
                win_offset.append([j,i])
        win_offset_torch = torch.tensor(win_offset,dtype=dtype, device=device)- win_size//2
    else:
        win_offset_torch = torch.tensor(offset,dtype=dtype, device=device)
    # print(win_offset_torch)
    # import pdb;pdb.set_trace()
    #* 为每个点添加偏置坐标
    for i in range(len(win_offset_torch)):
        win_xy_list.append((win_xy + win_offset_torch[i].view(2,1)).unsqueeze(2))
    win_xy_with_offset = torch.cat(win_xy_list, dim=2)  #* torch.Size([1, 2, 1024, 1280])
    win_x_normalized = win_xy_with_offset[:, 0, :, :] / ((width - 1) / 2) - 1
    win_y_normalized = win_xy_with_offset[:, 1, :, :] / ((height - 1) / 2) - 1
    win_grid = torch.stack((win_x_normalized, win_y_normalized), dim=3)  #* torch.Size([1, 1024, 1280, 2])
    # import pdb;pdb.set_trace()
    #* squared_gray_grad_patch.shape torch.Size([1, 1, 1024, 1280]) #* 总共1280个区域，每个区域共有1024个点，每个点一个平方灰度梯度
    grid_result = F.grid_sample(source, win_grid, mode="nearest", padding_mode="reflection", align_corners=True)
    return grid_result

def get_offset_grid(length=1,only4=False,show=False,scale = 1):
    rt_up_offset = [[i+1,0] for i in range(length)]
    rt_dw_offset = [[0,i+1] for i in range(length)]
    lt_up_offset = [[0,-i-1] for i in range(length)]
    lt_dw_offset = [[-i-1,0] for i in range(length)]
    if only4 and length==1:
        pass
    else:
        for j in range(length):
            for i in range(length):
                rt_up_offset.append([i+1,-j-1])
                rt_dw_offset.append([i+1,j+1])
                lt_up_offset.append([-i-1,-j-1])
                lt_dw_offset.append([-i-1,j+1])
    if show:
        all_offset_list = [rt_up_offset,rt_dw_offset,lt_dw_offset,lt_up_offset]
        plt_select_mask(all_offset_list,length,scale,name=f'latex/figures/detect_mask_len_{length}_scl_{scale}.png')
    #* 未排序，检查正确性：sparse_depth[0,0,500-7:501,501:508]  weight[0,0,0,:,500,500].view(8,7)
    #* 排序：
    rt_up_offset.sort(key=lambda p : abs(p[0])+abs(p[1]))
    rt_dw_offset.sort(key=lambda p : abs(p[0])+abs(p[1]))
    lt_up_offset.sort(key=lambda p : abs(p[0])+abs(p[1]))
    lt_dw_offset.sort(key=lambda p : abs(p[0])+abs(p[1]))
    return [rt_up_offset, rt_dw_offset, lt_dw_offset, lt_up_offset] #* 顺时针

def getcam_pts(depth, K=None,):
    if K is None :
        print("Invalid K ")
        return
    depth = depth.squeeze()
    if isinstance(depth,torch.Tensor):
        depth = depth.detach().float().cpu().numpy()
    if isinstance(K,torch.Tensor):
        K = K.detach().float().cpu().numpy()
    if len(depth.shape)>2:
        print("Invalid depth!")
        return
    height, width = depth.shape
    u, v = np.meshgrid(np.arange(0, width),np.arange(0, height))
    uv1 = np.stack([u.reshape(-1),v.reshape(-1),np.ones_like(v.reshape(-1))],axis=0) #* (3, 1310720)
    valid_depth_mask = depth>0
    cam_pts_tmp = np.linalg.inv(K) @ (uv1 * depth.reshape(-1))
    cam_pts = cam_pts_tmp.reshape(3, height, width)
    return cam_pts

def depth2ply(depth, image=None, K=None, T=None, return_pts = False, normal=None):
    if K is None or T is None:
        print("Invalid K or T")
        return
    depth = depth.squeeze()
    if isinstance(depth,torch.Tensor):
        depth = depth.detach().float().cpu().numpy()
    if isinstance(K,torch.Tensor):
        K = K.detach().float().cpu().numpy()
    if isinstance(T,torch.Tensor):
        T = T.detach().float().cpu().numpy()
    if isinstance(image,torch.Tensor):
        image = image.detach().float().cpu().numpy()
    if isinstance(normal,torch.Tensor):
        normal = normal.detach().float().cpu().numpy()
    if len(depth.shape)>2:
        print("Invalid depth!")
        return
    height, width = depth.shape
    u, v = np.meshgrid(np.arange(0, width),np.arange(0, height))
    # import pdb;pdb.set_trace()
    uv1 = np.stack([u.reshape(-1),v.reshape(-1),np.ones_like(v.reshape(-1))],axis=0) #* (3, 1310720)
    proj_w2c = T*1.0
    # proj_w2c = np.eye(4)
    proj_w2c[:3,:4] = K @ T[:3,:4]
    proj_c2w = np.linalg.inv(proj_w2c)
    rot = proj_c2w[:3, :3]  # [B,3,3]
    trans = proj_c2w[:3, 3:4]  # [B,3,1]
    #* TODO
    valid_depth_mask = (depth>0).reshape(-1)
    rot_depth_xyz = rot @ (uv1 * depth.reshape(-1))
    proj_xyz = rot_depth_xyz + trans
    color = np.ones_like(proj_xyz.T)
    if image is not None:
        color = image.reshape(-1,3)
        if color.max()<2:
            color = color*255
    pts = np.concatenate([proj_xyz.T, color],axis=1) #* (1310720, 6)
    #* TODO
    pts = pts[valid_depth_mask,:]
    if normal is not None:
        normal = normal.reshape(-1, 3)[valid_depth_mask,:]
        # rot = np.linalg.inv(T[:3,:3])
    if not return_pts:
        pcwrite("output/ply/depth.ply",pts, normal)
    else:
        return pts
    # import pdb;pdb.set_trace()

def pcwrite(filename, xyzrgb, normal=None):
    """Save a point cloud to a polygon .ply file.
    """
    xyz = xyzrgb[:, :3]
    rgb = xyzrgb[:, 3:].astype(np.uint8)
    # Write header
    ply_file = open(filename,'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n"%(xyz.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    if normal is not None and xyzrgb.shape[0]==normal.shape[0]:
        ply_file.write("property float nx\n")
        ply_file.write("property float ny\n")
        ply_file.write("property float nz\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("end_header\n")
    # Write vertex list
    for i in range(xyz.shape[0]):
        if normal is not None and xyzrgb.shape[0]==normal.shape[0]:
            ply_file.write("%f %f %f %f %f %f %d %d %d\n"%(
                xyz[i, 0], xyz[i, 1], xyz[i, 2],
                normal[i, 0], normal[i, 1], normal[i, 2],
                rgb[i, 0], rgb[i, 1], rgb[i, 2],
            ))
        else:
            ply_file.write("%f %f %f %d %d %d\n"%(
            xyz[i, 0], xyz[i, 1], xyz[i, 2],
            rgb[i, 0], rgb[i, 1], rgb[i, 2],
            ))

def to_plot_data(data): #* 转为可以用于绘图的数据
    if isinstance(data,torch.Tensor):
        data = data.detach().float().cpu().squeeze().numpy()
    if len(data.shape)==3 and data.shape[0]<data.shape[1] and data.shape[0]<data.shape[2] :
        data = data.transpose([1,2,0])
    if len(data.shape)>2 and (data.min()<0 or data.max()>1):
        data = (data-data.min())/(data.max()-data.min())
    if len(data.shape)>2 and data.shape[2]>3:
        data = data[:,:,:3]
    return data

def plot(data,show=False,name=None,cmap=None):
    if isinstance(data,torch.Tensor):
        data = data.detach().float().cpu().squeeze().numpy()
    if len(data.shape)==3 and data.shape[0]<=3:
        data = data.transpose([1,2,0])
    if len(data.shape)>2 and (data.min()<0 or data.max()>1):
        data = (data-data.min())/(data.max()-data.min())
    if len(data.shape)>2 and data.shape[2]>3:
        data = data[:,:,:3]
    plt.figure()
    if cmap is not None:
        plt.imshow(data,cmap)
    else:
        plt.imshow(data)
    if name is not None:
        plt.tight_layout()
        plt.axis("off")
        plt.savefig(name,dpi=500,bbox_inches='tight')
        plt.close('all') 
    if show:
        plt.show()

def plt_select_mask(all_offset_list,length,scale=1,show=False,name=None):
    num_offset = len(all_offset_list)
    color_values = np.arange(num_offset+2)
    select_mask = np.ones((2*length*scale+1,2*length*scale+1))*color_values[0]
    select_mask[length*scale,length*scale] = color_values[-1]
    for offset_list_i in range(num_offset):
        if isinstance(all_offset_list[offset_list_i][0],list):
            for uv in all_offset_list[offset_list_i]:
                # import pdb;pdb.set_trace()
                select_mask[length*scale+uv[1]*scale,length*scale+uv[0]*scale] = color_values[offset_list_i+1]
        else:
            uv = all_offset_list[offset_list_i]
            select_mask[length*scale+uv[1]*scale,length*scale+uv[0]*scale] = color_values[offset_list_i+1]
    plt.imshow(select_mask,extent=[-length*scale,length*scale+1,length*scale+1,-length*scale])
    for i in range(-length*scale,length*scale+1+1):
        plt.plot([i,i],[-length*scale,length*scale+1],'w')
        plt.plot([-length*scale,length*scale+1],[i,i],'w')
    plt.axis('off')
    # plt.grid()
    if name is not None:
        plt.tight_layout()
        plt.savefig(name,dpi=500,bbox_inches='tight')
        plt.close('all') 
    if show:
        plt.show()

def print_args(args: Any) -> None:
    """Utilities to print arguments

    Args:
        args: arguments to print out
    """
    print("################################  args  ################################")
    for k, v in args.__dict__.items():
        print("{0: <10}\t{1: <30}\t{2: <20}".format(k, str(v), str(type(v))))
    print("########################################################################")

def make_nograd_func(func: Callable) -> Callable:
    """Utilities to make function no gradient

    Args:
        func: input function

    Returns:
        no gradient function wrapper for input function
    """

    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            ret = func(*f_args, **f_kwargs)
        return ret

    return wrapper

def make_recursive_func(func: Callable) -> Callable:
    """Convert a function into recursive style to handle nested dict/list/tuple variables

    Args:
        func: input function

    Returns:
        recursive style function
    """

    def wrapper(args):
        if isinstance(args, list):
            return [wrapper(x) for x in args]
        elif isinstance(args, tuple):
            return tuple([wrapper(x) for x in args])
        elif isinstance(args, dict):
            return {k: wrapper(v) for k, v in args.items()}
        else:
            return func(args)

    return wrapper


@make_recursive_func
def tensor2float(args: Any) -> float:
    """Convert tensor to float"""
    if isinstance(args, float):
        return args
    elif isinstance(args, torch.Tensor):
        return args.data.item()
    else:
        raise NotImplementedError("invalid input type {} for tensor2float".format(type(args)))


@make_recursive_func
def tensor2numpy(args: Any) -> np.ndarray:
    """Convert tensor to numpy array"""
    if isinstance(args, np.ndarray):
        return args
    elif isinstance(args, torch.Tensor):
        return args.detach().cpu().numpy().copy()
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(args)))


@make_recursive_func
def to_cuda(args: Any) -> Union[str, torch.Tensor]:
    """Convert tensor to tensor on GPU"""
    if isinstance(args, torch.Tensor):
        return args.cuda().float()
    elif isinstance(args, str):
        return args
    else:
        raise NotImplementedError("invalid input type {} for to_cuda".format(type(args)))


def save_scalars(logger: SummaryWriter, mode: str, scalar_dict: Dict[str, Any], global_step: int) -> None:
    """Log values stored in the scalar dictionary

    Args:
        logger: tensorboard summary writer
        mode: mode name used in writing summaries
        scalar_dict: python dictionary stores the key and value pairs to be recorded
        global_step: step index where the logger should write
    """
    scalar_dict = tensor2float(scalar_dict)
    for key, value in scalar_dict.items():
        if not isinstance(value, (list, tuple)):
            name = "{}/{}".format(mode, key)
            logger.add_scalar(name, value, global_step)
        else:
            for idx in range(len(value)):
                name = "{}/{}_{}".format(mode, key, idx)
                logger.add_scalar(name, value[idx], global_step)


def save_images(logger: SummaryWriter, mode: str, images: Dict[str, np.ndarray], global_step: int) -> None:
    """Log images stored in the image dictionary

    Args:
        logger: tensorboard summary writer
        mode: mode name used in writing summaries
        images: python dictionary stores the key and image pairs to be recorded
        global_step: step index where the logger should write
    """
    def preprocess(image_name, image):
        if not (len(image.shape) == 3 or len(image.shape) == 4):
            raise NotImplementedError("invalid img shape {}:{} in save_images".format(image_name, image.shape))
        if len(image.shape) == 3:
            image = image[:, np.newaxis, :, :]
        image = torch.from_numpy(image[:1])
        return torchvision.utils.make_grid(image, padding=0, nrow=1, normalize=True, scale_each=True)

    for key, value in images.items():
        if not isinstance(value, (list, tuple)):
            name = "{}/{}".format(mode, key)
            logger.add_image(name, preprocess(name, value), global_step)
        else:
            for idx in range(len(value)):
                name = "{}/{}_{}".format(mode, key, idx)
                logger.add_image(name, preprocess(name, value[idx]), global_step)


class DictAverageMeter:
    """Wrapper class for dictionary variables that require the average value"""

    def __init__(self) -> None:
        """Initialization method"""
        self.data: Dict[Any, float] = {}
        self.count = 0

    def update(self, new_input: Dict[Any, float]) -> None:
        """Update the stored dictionary with new input data

        Args:
            new_input: new data to update self.data
        """
        self.count += 1
        if len(self.data) == 0:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.data[k] = v
        else:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.data[k] += v

    def mean(self) -> Any:
        """Return the average value of values stored in self.data"""
        return {k: v / self.count for k, v in self.data.items()}


def compute_metrics_for_each_image(metric_func: Callable) -> Callable:
    """A wrapper to compute metrics for each image individually"""

    def wrapper(depth_est, depth_gt, mask, *args):
        batch_size = depth_gt.shape[0]
        results = []
        # compute result one by one
        for idx in range(batch_size):
            ret = metric_func(depth_est[idx], depth_gt[idx], mask[idx], *args)
            results.append(ret)
        return torch.stack(results).mean()

    return wrapper


@make_nograd_func
@compute_metrics_for_each_image
def threshold_metrics(
    depth_est: torch.Tensor, depth_gt: torch.Tensor, mask: torch.Tensor, threshold: float
) -> torch.Tensor:
    """Return error rate for where absolute error is larger than threshold.

    Args:
        depth_est: estimated depth map
        depth_gt: ground truth depth map
        mask: mask
        threshold: threshold

    Returns:
        error rate: error rate of the depth map
    """
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    errors = torch.abs(depth_est - depth_gt).float()
    err_mask = errors > threshold
    return torch.mean(err_mask.float())


# NOTE: please do not use this to build up training loss
@make_nograd_func
@compute_metrics_for_each_image
def absolute_depth_error_metrics(depth_est: torch.Tensor, depth_gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Calculate average absolute depth error

    Args:
        depth_est: estimated depth map
        depth_gt: ground truth depth map
        mask: mask
    """
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    return torch.mean((depth_est - depth_gt).abs())


def test_optical_flow_dense():
    import numpy as np
    import cv2 as cv
    cap = cv.VideoCapture('/home/zhujun/MVS/PatchmatchNet/data/vtest.avi')
    ret, frame1 = cap.read()
    prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    while(1):
        ret, frame2 = cap.read()
        if not ret:
            print('No frames grabbed!')
            break

        next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        cv.imshow('frame2', bgr)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv.imwrite('opticalfb.png', frame2)
            cv.imwrite('opticalhsv.png', bgr)
        prvs = next

    cv.destroyAllWindows()

def test_optical_flow():
    import numpy as np
    import cv2 as cv
    import argparse

    parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
                                                The example file can be downloaded from: \
                                                https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')

    cap = cv.VideoCapture('/home/zhujun/MVS/PatchmatchNet/data/vtest.avi')

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                        qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15, 15),
                    maxLevel = 2,
                    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    while(1):
        ret, frame = cap.read()
        if not ret:
            print('No frames grabbed!')
            break

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        img = cv.add(frame, mask)

        cv.imshow('frame', img)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cv.destroyAllWindows()

if __name__=="__main__":
    # test_optical_flow_dense()
    test_optical_flow()