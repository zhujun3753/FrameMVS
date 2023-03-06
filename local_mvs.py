import os
os.environ['NUMEXPR_MAX_THREADS'] = "24"
import argparse
import cv2
from isort import file
from matplotlib import image
import numpy as np
import sys
import time
from requests import delete
from torch import RRefType
import torch.nn as nn
from typing import Tuple
from torch.utils.data import DataLoader
import matplotlib as mpl


from datasets.data_io import read_cam_file, read_image, read_map, read_pair_file, save_image, save_map
# from datasets.mvs import MVSDataset,R3liveDataset
from torch.utils.data import Dataset
# from models.net import PatchmatchNet,patchmatchnet_loss

from utils import print_args, tensor2numpy, to_cuda
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import List
import torch.nn.functional as F
from utils import plot
from tsdf_fusion.sparse_volume import SparseVolume
from tsdf_fusion.fusion import *
import copy
import random
import trimesh
import open3d as o3d
from config import config as cfg
from utils import *
from models.module import differentiable_warping,proj_ref_src
from image_cpp_cuda_tool import images_seg
from ACMMP_net import *
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

class LocalMVSDataset(Dataset):
    #* 专门用于读取原始数据，利用原始数据得到局部MVS所需数据
    def __init__(self,config):
        super(LocalMVSDataset, self).__init__()
        self.data_path = config.data_path
        self.num_views = config.num_views
        self.max_dim = config.max_dim
        self.robust_train = config.robust_train
        self.image_extension = config.image_extension
        image_list = [f for f in os.listdir(os.path.join(config.data_path, config.depth_folder)) if self.image_extension in f]
        image_list.sort(key=lambda f: int(f.split('.')[0]))
        self.image_filenames = [ os.path.join(config.data_path, config.image_folder,fn)  for fn in image_list]
        self.depth_filenames = [ os.path.join(config.data_path, config.depth_folder,fn)  for fn in image_list]
        self.ext_filenames = [ os.path.join(config.data_path, config.extrinsic_folder,fn.replace('.png','.txt'))  for fn in image_list]
        K_path =os.path.join(config.data_path, config.intrinsic_folder, 'intrinsic.txt')
        self.K = np.loadtxt(K_path)
        self.last_idx = 0
        self.cur_idx = 0
        self.marginal_idx = 0
        self.exts = [np.loadtxt(ext_filename) for ext_filename in self.ext_filenames]
        self.poses = [np.linalg.inv(ext) for ext in self.exts]
        self.history_key_idx = []
        self.sliding_win_ids = []
        self.sliding_win_len = config.sliding_win_len
        self.K_torch = torch.from_numpy(self.K).cuda()
        self.K_inv_torch = torch.inverse(self.K_torch)
        self.T_cws_torch = [torch.from_numpy(ext).cuda() for ext in self.exts]
        self.T_wcs_torch = [torch.from_numpy(pose).cuda() for pose in self.poses]
        self.set_win_grid()
        self.sw_images  = []
        self.sw_depths  = []
        self.sw_exts  = []
        self.sw_ints  = []

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        return cv2.imread(self.image_filenames[idx])
    
    def set_win_grid(self):
        dtype=self.K_torch.dtype
        device=self.K_torch.device
        cur_depth = cv2.imread(self.depth_filenames[self.cur_idx], -1)/1000.0
        win_size = 32
        batch = 1
        height,width = cur_depth.shape
        win_y_grid, win_x_grid = torch.meshgrid([torch.arange(win_size//2, height, win_size, dtype=dtype, device=device), torch.arange(win_size//2, width, win_size, dtype=dtype, device=device),])
        new_h,new_w = win_y_grid.shape
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
        self.win_xy_with_offset = torch.cat(win_xy_list, dim=2)  #* torch.Size([1, 2, 1024, 1280])
        win_x_normalized = self.win_xy_with_offset[:, 0, :, :] / ((width - 1) / 2) - 1
        win_y_normalized = self.win_xy_with_offset[:, 1, :, :] / ((height - 1) / 2) - 1
        self.win_grid = torch.stack((win_x_normalized, win_y_normalized), dim=3)  #* torch.Size([1, 1024, 1280, 2])

    def clear_data(self):
        self.history_key_idx =[]
        self.sliding_win_ids =[]
        self.cur_idx=0
    
    def get_next_data(self,ths=0.1, invalid_ratio_ths = 0.4):
        #* 借鉴DSO的方式，滑窗MVS, 左3帧和右3帧优化中间帧
        find_new_Key_frame = True
        if len(self.sliding_win_ids)==0: #* 第一帧，需要先填满滑窗
            self.history_key_idx.append(0)
            self.sliding_win_ids.append(0)
            self.sw_images.append(cv2.imread(self.image_filenames[0]))
            self.sw_exts.append(self.exts[0])
            self.sw_ints.append(self.K)
        elif len(self.sliding_win_ids) == self.sliding_win_len:
            self.marginal_idx, self.sliding_win_ids = self.sliding_win_ids[0], self.sliding_win_ids[1:]
            self.sw_images  = self.sw_images[1:]
            self.sw_exts  = self.sw_exts[1:]
            self.sw_ints  = self.sw_ints[1:]
        while find_new_Key_frame and len(self.sliding_win_ids) < self.sliding_win_len:
            self.cur_idx = self.sliding_win_ids[-1]
            #* 深度应该是临时渲染出来的
            cur_depth = cv2.imread(self.depth_filenames[self.cur_idx], -1)/1000.0
            T_wc = self.T_wcs_torch[self.cur_idx]
            dtype=self.K_torch.dtype
            device=self.K_torch.device
            depth_torch = torch.from_numpy(cur_depth).to(device)[None,None] #* torch.Size([1, 1, 1024, 1280])
            height,width = cur_depth.shape
            depth_sample = F.grid_sample(depth_torch, self.win_grid, mode="nearest", padding_mode="reflection", align_corners=True)
            indices = depth_sample.max(dim=2).indices.squeeze().cpu().numpy()
            depths = depth_sample.max(dim=2).values.squeeze() #* torch.Size([1280])
            depth_uv = [self.win_xy_with_offset[0,:,indices[j],j] for j in range(self.win_xy_with_offset.shape[-1])]
            depth_uv_torch = torch.stack(depth_uv,dim=0) #* torch.Size([1280, 2])
            depth_uv1_torch = torch.cat([depth_uv_torch,torch.ones_like(depth_uv_torch[:,0:1])],dim=1) #* torch.Size([1280, 3])
            depth_uvd_torch = depth_uv1_torch*depths.unsqueeze(1)
            depth_uvd_torch_valid = depth_uvd_torch[depths>0,:] #* torch.Size([1280, 3])
            depth_xyz_c = torch.matmul(self.K_inv_torch,depth_uvd_torch_valid.T) #* torch.Size([3, 1280])
            depth_xyz1_w = torch.matmul(T_wc,torch.cat([depth_xyz_c,torch.ones_like(depth_xyz_c[0:1,:])],dim=0)) #* torch.Size([4, 1280])
            depth_xyz_w = depth_xyz1_w[:3,:]
            find_new_Key_frame = False
            for idx in range(self.cur_idx + 1, len(self.image_filenames)):
                proj = self.T_cws_torch[idx]*1.0
                proj[:3,:] = torch.matmul(self.K_torch,proj[:3,:])
                depth_uvd1_cj = torch.matmul(proj,depth_xyz1_w)
                depth_uv_cj = depth_uvd1_cj[:2]/(depth_uvd1_cj[2]+1e-6)
                valid_mask = (depth_uvd1_cj[2,:]>0.1)*(depth_uv_cj[0]>0)*(depth_uv_cj[0]<width)*(depth_uv_cj[1]>0)*(depth_uv_cj[1]<height) #* torch.Size([1280])
                invalid_ratio = 1.0 - valid_mask.sum().cpu().numpy() / len(valid_mask)
                # print(valid_mask.shape,valid_mask.sum().cpu().numpy())
                selected_pts = depth_xyz_w[:,valid_mask] #* torch.Size([3, 772])
                #* vector to both camera center
                v1 = self.T_wcs_torch[idx][:3,3:4] - selected_pts
                v2 = self.T_wcs_torch[self.cur_idx][:3,3:4] - selected_pts #* torch.Size([3, 871])
                distance = (self.T_wcs_torch[idx][:3,3:4]-self.T_wcs_torch[self.cur_idx][:3,3:4]).norm() #* cm
                # distance = torch.clamp(distance,0,300)
                # import pdb;pdb.set_trace()
                cos_thetas = (v1*v2).sum(dim=0)/(v1.norm(dim=0)*v2.norm(dim=0)) #* torch.Size([871])
                thetas = torch.arccos(cos_thetas) #* (180 / torch.pi)*
                invalid_mask = torch.isnan(thetas)
                valid_theta = thetas[~invalid_mask]
                valid_theta = valid_theta[valid_theta<torch.pi/3]
                view_score = np.sqrt(np.sqrt(len(valid_theta)))*valid_theta.sum()/1000.0 * torch.exp(-torch.abs(distance-10)/10)
                view_score_np = view_score.cpu().numpy()
                if view_score_np>ths or invalid_ratio > invalid_ratio_ths:
                    # print(self.cur_idx, idx, view_score_np)
                    self.history_key_idx.append(idx)
                    self.sliding_win_ids.append(idx)
                    self.sw_images.append(cv2.imread(self.image_filenames[idx]))
                    self.sw_exts.append(self.exts[idx])
                    self.sw_ints.append(self.K)
                    find_new_Key_frame = True
                    break
                # if idx>400:
                #     import pdb;pdb.set_trace()
        if find_new_Key_frame:
            # print(self.sliding_win_ids)
            ref_idx = self.sliding_win_len//2
            indices = self.sliding_win_ids[ref_idx:ref_idx+1] + self.sliding_win_ids[:ref_idx] + self.sliding_win_ids[ref_idx+1:]
            images = self.sw_images[ref_idx:ref_idx+1] + self.sw_images[:ref_idx] + self.sw_images[ref_idx+1:]
            exts = self.sw_exts[ref_idx:ref_idx+1] + self.sw_exts[:ref_idx] + self.sw_exts[ref_idx+1:]
            ints = self.sw_ints[ref_idx:ref_idx+1] + self.sw_ints[:ref_idx] + self.sw_ints[ref_idx+1:]
            # import pdb;pdb.set_trace()
            #* 临时投影
            depth = cv2.imread(self.depth_filenames[self.sliding_win_ids[ref_idx]], -1)/1000.0
            indices_torch = torch.tensor(indices).float().cuda()
            images_torch = torch.from_numpy(np.stack(images,axis=0)).permute(0,3,1,2).float().cuda()[:,[2,1,0],:,:]
            ext_torch = torch.from_numpy(np.stack(exts, axis=0)).float().cuda()[None,  ...]
            ints_torch = torch.from_numpy(np.stack(ints, axis=0)).float().cuda()[None,  ...]
            #* 临时投影
            # depth = self.sparse_volume.proj2depth(ext_torch[0,0], ints_torch[0,0], images_torch.shape[-1], images_torch.shape[-2])
            # import pdb;pdb.set_trace()
            depth_torch = torch.from_numpy(depth)[None, None, ...].float().cuda()
            data = {
                "indices": indices_torch,
                "images": images_torch, #* (N, 3, H, W)
                "depth": depth_torch, #* torch.Size([1, 1,  1024, 1280])
                "exts": ext_torch, #* (1, N, 4, 4)
                "ints": ints_torch, #* (1, N, 3, 3)
            }
            return data
        else:
            return None

    def get_next_data_all(self):
        if len(self.image_filenames)<=self.cur_idx:
            return None
        exts = [self.exts[self.cur_idx]]
        ints = [self.K]
        indices_torch = torch.tensor([self.cur_idx]).float().cuda()
        # import pdb;pdb.set_trace()
        images_torch = torch.from_numpy(np.stack([cv2.imread(self.image_filenames[self.cur_idx])[:,:,[2,1,0]]],axis=0)).permute(0,3,1,2).float().cuda()
        ext_torch = torch.from_numpy(np.stack(exts, axis=0)).float().cuda()[None,  ...]
        ints_torch = torch.from_numpy(np.stack(ints, axis=0)).float().cuda()[None,  ...]
        #* 临时投影
        depth = cv2.imread(self.depth_filenames[self.cur_idx], -1)/1000.0
        # import pdb;pdb.set_trace()
        depth_torch = torch.from_numpy(depth)[None, None, ...].float().cuda()
        data = {
            "indices": indices_torch,
            "images": images_torch, #* (N, 3, H, W)
            "depth": depth_torch, #* torch.Size([1, 1,  1024, 1280])
            "exts": ext_torch, #* (1, N, 4, 4)
            "ints": ints_torch, #* (1, N, 3, 3)
        }
        self.cur_idx += 1
        return data

    def get_cur_data(self):
        ref_idx = self.sliding_win_len//2
        indices = self.sliding_win_ids[ref_idx:ref_idx+1] + self.sliding_win_ids[:ref_idx] + self.sliding_win_ids[ref_idx+1:]
        print(indices)
        images = self.sw_images[ref_idx:ref_idx+1] + self.sw_images[:ref_idx] + self.sw_images[ref_idx+1:]
        exts = self.sw_exts[ref_idx:ref_idx+1] + self.sw_exts[:ref_idx] + self.sw_exts[ref_idx+1:]
        ints = self.sw_ints[ref_idx:ref_idx+1] + self.sw_ints[:ref_idx] + self.sw_ints[ref_idx+1:]
        # import pdb;pdb.set_trace()
        #* 临时投影
        # depth = cv2.imread(self.depth_filenames[self.sliding_win_ids[ref_idx]], -1)/1000.0
        indices_torch = torch.tensor(indices).float().cuda()
        images_torch = torch.from_numpy(np.stack(images,axis=0)).permute(0,3,1,2).float().cuda()
        ext_torch = torch.from_numpy(np.stack(exts, axis=0)).float().cuda()[None,  ...]
        ints_torch = torch.from_numpy(np.stack(ints, axis=0)).float().cuda()[None,  ...]
        #* 临时投影
        depth = self.sparse_volume.proj2depth(ext_torch[0,0], ints_torch[0,0], images_torch.shape[-1], images_torch.shape[-2])
        # import pdb;pdb.set_trace()
        depth_torch = torch.from_numpy(depth)[None, None, ...].float().cuda()
        data = {
            "indices": indices_torch,
            "images": images_torch, #* (N, 3, H, W)
            "depth": depth_torch, #* torch.Size([1, 1,  1024, 1280])
            "exts": ext_torch, #* (1, N, 4, 4)
            "ints": ints_torch, #* (1, N, 3, 3)
        }
        return data


class LocalMVS():
    def __init__(self,config):
        super(LocalMVS, self).__init__()
        self.cfg = config
        #* 原始数据集,需要什么原始数据都在这里面找
        self.local_mvs_dataset = LocalMVSDataset(config)
        #* sparse_volume
        self.reset_sparse_volume()
        #* mvs index
        self.mvs_indices = []
        #* 有效范围
        self.lidar_range = config.lidar_range
        self.mvs_range = config.mvs_range
    
    def reset_sparse_volume(self,):
        #* 读取ply数据保存到np中
        # ply_np = read_ply(os.path.join(self.cfg.data_path, self.cfg.ply_filename)) #* (8143478, 6) xyzrgb
        # ply_np = read_ply("/home/zhujun/MVS/FrameMVS/output/ply/lidar_pcd.ply") #* (8143478, 6) xyzrgb
        # xyz_min = ply_np[:,:3].min()
        # xyz_max = ply_np[:,:3].max()
        # bound_max = 2*max(abs(xyz_min),abs(xyz_max))
        # dimensions=np.asarray([bound_max, bound_max, bound_max])
        #* 将np数据加入到hashmap中
        # Global_map.append_points_to_global_map(torch.from_numpy(ply_np).clone().float())
        Global_map.cur_resolution()
        # test_pc = Global_map.get_pc()
        # test_pc_np = test_pc.numpy()
        # pcwrite("output/ply/test_pc_np.ply", test_pc_np)
        # import pdb;pdb.set_trace()
        # 
        # self.sparse_volume = SparseVolume(n_feats=self.cfg.n_feats, voxel_size=self.cfg.voxel_size, dimensions=dimensions, device="cpu:0")
        # self.sparse_volume.integrate_pts(ply_np) #* sparse_volume.size() 6152961
        # import pdb;pdb.set_trace()
        # test_pts=self.sparse_volume.to_pc()
        # pcwrite(f"output/ply/xyzrgb_{3}.ply", test_pts)

    def diff_cost_test(self, depth_test, cost_test, images_tensor, intrinsics, extrinsics):
        #* 当前的代价并不能有效滤除无效点！！！！
        depth_test_orig = depth_test*1.0
        ths = np.linspace(0, 1, 5)
        for th in ths:
            depth_test = depth_test_orig * 1.0
            invalid_mask = cost_test>th
            depth_test[invalid_mask]=0.0
            acmmp_pts = depth2ply(depth_test, image=images_tensor[0].permute(1,2,0)[:,:,[2,1,0]], K=intrinsics[0,0], T=extrinsics[0][0],return_pts = True)
            pcd1 = o3d.geometry.PointCloud()
            pcd1.points = o3d.utility.Vector3dVector(acmmp_pts[:,:3])
            pcd1.colors = o3d.utility.Vector3dVector(acmmp_pts[:,3:6]/255.0)
            o3d.visualization.draw_geometries([pcd1])
            print(th)
            # o3d.io.write_point_cloud("output/ply/test_comb.ply", pcd1)
        # o3d.io.write_point_cloud("output/ply/test_comb.ply", pcd1.transform(new_T)+pcd2)
        import pdb;pdb.set_trace()

    def fusion(self):
        #? 获取当前局部MVS数据,cuda
        iter_num = 0
        start = time.time()
        stop = False
        Global_map.set_resolution(0.01,0.1)
        self.o3d_tsdf_fusion = TSDFFusion(voxel_length=0.02, sdf_trunc=0.2)
        while(~stop):
            print("iter_num: ",iter_num)
            if iter_num%10==0 and iter_num>0:
                # xyzrgb = self.sparse_volume.to_pc()
                # pcwrite(f"output/ply/xyzrgb_{iter_num}.ply", xyzrgb)
                write_flag = False
                # import pdb;pdb.set_trace()
                if write_flag:
                    file_path =  os.path.join('output', 'ply', f'o3d{iter_num}.ply')
                    if not os.path.exists(file_path):
                        self.o3d_tsdf_fusion.marching_cube(path=file_path)
                    # test_pc = Global_map.get_pc()
                    # pcwrite(f"output/ply/test_pc_np_{iter_num}.ply", test_pc.numpy())
            sample_data = self.local_mvs_dataset.get_next_data_all()
            if sample_data is None:
                break
            iter_num+=1
            indices = sample_data['indices'].cpu().numpy().astype(int) #* [ref_idx, src_idx1, ...]
            print(indices)
            images_tensor = sample_data['images'] #* torch.Size([3, 3, 1024, 1280])
            intrinsics = sample_data['ints'] #* torch.Size([1, 3, 3, 3])
            extrinsics = sample_data['exts'] #* torch.Size([1, 3, 4, 4])
            orig_depth = sample_data['depth'] #* torch.Size([1, 1, 1024, 1280])
            #? 初步稀疏深度补全，局部线性插值，置信度较高
            sparse2comp_bi = sparse_depth_to_dense(orig_depth).unsqueeze(1)
            sample_data['sparse2comp_bi'] = sparse2comp_bi
            init_depth = sparse2comp_bi #* torch.Size([1, 1, 1024, 1280])
            # init_depth = orig_depth #* torch.Size([1, 1, 1024, 1280])
            # invalid_mask = (init_depth>40) | (init_depth<2.5)
            invalid_mask = (init_depth>40)
            init_depth[invalid_mask] = 0.0
            filter_depth = simpletools.depth_filter(init_depth.squeeze(0).cpu().float())
            init_depth = filter_depth.unsqueeze(0).unsqueeze(0)

            #? 生成点云测试
            depth_np = init_depth.squeeze().float().cpu().numpy()
            rgb = images_tensor[0].permute(1,2,0).cpu().numpy()
            pose = np.linalg.inv(extrinsics[0][0].cpu().numpy())
            K = intrinsics[0,0].cpu().numpy()
            self.o3d_tsdf_fusion.integrate(depth_np, rgb, pose, K)
            # o3d_tsdf_fusion_single = TSDFFusion(0.02)
            # o3d_tsdf_fusion_single.integrate(depth_np, rgb, pose, K)
            # mesh_o3d = o3d_tsdf_fusion_single.marching_cube(path=os.path.join('output', 'ply', f'o3d_single_{iter_num}.ply'))
            # import pdb;pdb.set_trace()
            print("cost time per iter: ", time.time() - start, "s")
            start = time.time()
        self.final_mesh = self.o3d_tsdf_fusion.marching_cube(path=os.path.join('output', 'ply', 'o3d_final.ply'))
        # import pdb;pdb.set_trace()


    def simple_comp(self):
        #? 获取当前局部MVS数据,cuda
        iter_num = 0
        start = time.time()
        stop = False
        Global_map.set_resolution(0.01,0.1)
        self.o3d_tsdf_fusion = TSDFFusion(voxel_length=0.02)
        while(~stop):
            print("iter_num: ",iter_num)
            if iter_num%10==0 and iter_num>0:
                # xyzrgb = self.sparse_volume.to_pc()
                # pcwrite(f"output/ply/xyzrgb_{iter_num}.ply", xyzrgb)
                write_flag = False
                # import pdb;pdb.set_trace()
                if write_flag:
                    file_path =  os.path.join('output', 'ply', f'o3d{iter_num}.ply')
                    if not os.path.exists(file_path):
                        self.o3d_tsdf_fusion.marching_cube(path=file_path)
                    # test_pc = Global_map.get_pc()
                    # pcwrite(f"output/ply/test_pc_np_{iter_num}.ply", test_pc.numpy())
            sample_data = self.local_mvs_dataset.get_next_data()
            if sample_data is None:
                break
            iter_num+=1
            indices = sample_data['indices'].cpu().numpy().astype(int) #* [ref_idx, src_idx1, ...]
            print(indices)
            images_tensor = sample_data['images'] #* torch.Size([3, 3, 1024, 1280])
            intrinsics = sample_data['ints'] #* torch.Size([1, 3, 3, 3])
            extrinsics = sample_data['exts'] #* torch.Size([1, 3, 4, 4])
            orig_depth = sample_data['depth'] #* torch.Size([1, 1, 1024, 1280])
            #? 初步稀疏深度补全，局部线性插值，置信度较高
            sparse2comp_bi = sparse_depth_to_dense(orig_depth).unsqueeze(1)
            sample_data['sparse2comp_bi'] = sparse2comp_bi
            init_depth = sparse2comp_bi #* torch.Size([1, 1, 1024, 1280])
            # init_depth = orig_depth #* torch.Size([1, 1, 1024, 1280])
            # invalid_mask = (init_depth>40) | (init_depth<2.5)
            invalid_mask = (init_depth>40)
            init_depth[invalid_mask] = 0.0
            filter_depth = simpletools.depth_filter(init_depth.squeeze(0).cpu().float())
            init_depth = filter_depth.unsqueeze(0).unsqueeze(0)
            # images_plot=[]
            # images_plot.append(['init_depth', init_depth.squeeze().float().cpu().numpy(),'jet'])
            # images_plot.append(['filter_depth', filter_depth.squeeze().cpu().numpy(),'jet'])
            # plt_imgs(images_plot,show=True,save=False)
            # import pdb;pdb.set_trace()
            # #? 轮廓提取
            edge_maps_tensor = images_seg.get_edge_maps_tensor(images_tensor) #* torch.Size([3, 1, 1024, 1280])
            torch.cuda.empty_cache()
            # #? 基于轮廓的区域分割
            seg_masks_tensor = images_seg.get_region_seg_tensor(edge_maps_tensor)
            torch.cuda.empty_cache()
            def plot_test(ths=0.5,save=False):
                seg_masks_tensor = images_seg.get_region_seg_tensor(edge_maps_tensor, ths) #* torch.Size([3, 1024, 1280])
                images_plot=[]
                images_plot.append(['images_tensor', images_tensor[0].permute(1,2,0).squeeze().cpu().numpy()/255.0,'jet'])
                images_plot.append(['edge_maps_tensor', edge_maps_tensor[0].squeeze().cpu().numpy(),'jet'])
                # segmask_np = shuffle_mask(seg_masks_tensor[0].squeeze().cpu().numpy())
                segmask_np = seg_masks_tensor[0].squeeze().cpu().numpy()
                images_plot.append(['seg_masks_tensor', segmask_np,'jet'])
                images_plot.append(['init_depth', init_depth.squeeze().float().cpu().numpy(),'jet'])
                plt_imgs(images_plot,show=True,save=save)
            ths = 0.01
            # import pdb;pdb.set_trace()
            seg_masks_tensor = images_seg.get_region_seg_tensor(edge_maps_tensor, ths) #* torch.Size([3, 1024, 1280])
            torch.cuda.empty_cache()
            # import pdb;pdb.set_trace()
            # images_plot=[]
            # images_plot.append(['images_tensor', images_tensor[0].permute(1,2,0).squeeze().cpu().numpy()/255.0,'jet'])
            # images_plot.append(['edge_maps_tensor', edge_maps_tensor[0].squeeze().cpu().numpy(),'jet'])
            # # segmask_np = shuffle_mask(seg_masks_tensor[0].squeeze().cpu().numpy())
            segmask_np = seg_masks_tensor[0].squeeze().cpu().numpy()
            # images_plot.append(['seg_masks_tensor', segmask_np,'jet'])
            # images_plot.append(['init_depth', init_depth.squeeze().float().cpu().numpy(),'jet'])
            # plt_imgs(images_plot,show=True,save=False)
            segmask_np_test = segmask_np*1.0
            invalid_mask = segmask_np_test!=771843.0
            segmask_np_test[invalid_mask] = -1
            plot(invalid_mask,show=True,name="output/ply/selected_region.png")
            depth_np = init_depth.squeeze().float().cpu().numpy()
            rgb = images_tensor[0].permute(1,2,0).cpu().numpy()
            rgb[~invalid_mask,:] = [255,0,0]
            pose = np.linalg.inv(extrinsics[0][0].cpu().numpy())
            K = intrinsics[0,0].cpu().numpy()
            self.o3d_tsdf_fusion.integrate(depth_np, rgb, pose, K)
            o3d_tsdf_fusion_single = TSDFFusion(0.02)
            o3d_tsdf_fusion_single.integrate(depth_np, rgb, pose, K)
            mesh_o3d = o3d_tsdf_fusion_single.marching_cube(path=os.path.join('output', 'ply', f'o3d_single_{iter_num}.ply'))
            height, width = depth_np.shape
            
            render = o3d.visualization.rendering.OffscreenRenderer(width, height)
            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = 'defaultLit'
            render.scene.add_geometry("mesh_o3d", mesh_o3d, mat)
            render.setup_camera(K, extrinsics[0][0].cpu().numpy(), width, height)
            cimg = np.asarray(render.render_to_image()) #* (1024, 1280, 3)
            dimg = np.asarray(render.render_to_depth_image())
            z_near = render.scene.camera.get_near()
            z_far = render.scene.camera.get_far()
            depth_render = 2.0 * z_near * z_far / (z_far + z_near - (2.0 * dimg - 1.0) * (z_far - z_near))
            depth_render[depth_render==depth_render.max()] = 0.0 #* (1024, 1280)
            images_plot=[]
            images_plot.append(['images_tensor', images_tensor[0].permute(1,2,0).squeeze().cpu().numpy()/255.0,'jet'])
            images_plot.append(['edge_maps_tensor', edge_maps_tensor[0].squeeze().cpu().numpy(),'jet'])
            # segmask_np = shuffle_mask(seg_masks_tensor[0].squeeze().cpu().numpy())
            segmask_np = seg_masks_tensor[0].squeeze().cpu().numpy()
            images_plot.append(['seg_masks_tensor', segmask_np,'jet'])
            images_plot.append(['init_depth', init_depth.squeeze().float().cpu().numpy(),'jet'])
            images_plot.append(['cimg', cimg,'jet'])
            images_plot.append(['depth_render', depth_render,'jet'])
            # plt_imgs(images_plot,show=True,save=False)
            #* 将法线信息对应到每个像素上，c++实现
            vertices=np.asarray(mesh_o3d.vertices) #* (422399, 3)
            vertex_normals=np.asarray(mesh_o3d.vertex_normals) #* (422399, 3)
            #* 转换为uvd
            ext = extrinsics[0][0].cpu().numpy()
            pts_cam = ext[:3,:3] @ vertices.T + ext[:3,3:] #* (3, 422399)
            uvd = K @ pts_cam
            uvd[:2] /= (uvd[2:]+1e-5)
            uvd_normal = np.concatenate([uvd.T,vertex_normals], axis=1)
            uvd_normal_torch = torch.from_numpy(uvd_normal) #* torch.Size([422399, 6])
            # import pdb;pdb.set_trace()
            depth_normal_torch = simpletools.proj_depth_normal(uvd_normal_torch.float(), width, height)
            depth_normal_torch[depth_normal_torch==1000]=0
            depth_proj = depth_normal_torch[:,:,0].cpu().numpy()
            normal_proj = depth_normal_torch[:,:,1:].cpu().numpy()
            depth_for_comp = depth_render * 1.0
            #* 相机坐标系下的三维点计算，使用这个坐标系是希望法线指向相机坐标系原点方向
            cam_pts = getcam_pts(depth_for_comp, K) #* (3, 1024, 1280)
            #* torch.Size([8, 1024, 1280]) [r,g,b,e,d，normal] 颜色3、颜色平滑度1、深度1, 法线3, 相机坐标系下三维坐标3
            image_info_torch = torch.cat([
                images_tensor[0].cpu(),  #* 3 rgb
                edge_maps_tensor[0].cpu(),  #* 1 edge
                torch.from_numpy(depth_for_comp).unsqueeze(0), #* 1 depth
                # init_depth.squeeze(0).cpu(), #* 1 depth
                depth_normal_torch[:,:,1:].cpu().permute(2,0,1), #* 3 normal 当前法线还比较稀疏，后续可以再重新算一下
                torch.from_numpy(cam_pts).float(), #* 3 pts
                ])
            normal_proj_plot = (normal_proj+1)*0.5
            images_plot=[]
            images_plot.append(['images_tensor', images_tensor[0].permute(1,2,0).squeeze().cpu().numpy()/255.0,'jet'])
            images_plot.append(['edge_maps_tensor', edge_maps_tensor[0].squeeze().cpu().numpy(),'jet'])
            # images_plot.append(['seg_masks_tensor', segmask_np,'jet']) #* shuffle_mask
            images_plot.append(['seg_masks_tensor', shuffle_mask(segmask_np),'jet']) #* shuffle_mask
            # images_plot.append(['init_depth', init_depth.squeeze().float().cpu().numpy(),'jet'])
            images_plot.append(['depth_for_comp', depth_for_comp,'jet'])
            # images_plot.append(['depth_normal_torch', normal_proj,'jet'])
            # images_plot.append(['depth_render', depth_render,'jet'])
            # images_plot.append(['edge_maps_tensor', edge_maps_tensor[0].squeeze().cpu().numpy(),'jet'])
            # plt_imgs(images_plot,show=True,save=False)
            # lidar_pcd = o3d.geometry.PointCloud()
            # lidar_pcd.points = mesh_o3d.vertices
            # o3d.visualization.draw_geometries([lidar_pcd])
            # print("cam_pts[:,901:905,1122]: ",cam_pts[:,901:905,1122])
            cam_param_torch = torch.tensor([K[0,0], K[0,2], K[1,1], K[1,2]])
            # import pdb;pdb.set_trace()
            result = simpletools.depth_interp_comp(image_info_torch, cam_param_torch)
            depth_test = result[:,:,0].cpu().numpy()
            curve_test = result[:,:,4].cpu().numpy()
            images_plot.append(['curve_test', curve_test,'jet'])
            images_plot.append(['depth_test', depth_test,'jet'])
            sparse_pts = depth2ply(depth_for_comp, image=images_tensor[0].permute(1,2,0), K=intrinsics[0,0], T=np.eye(4),return_pts = True)
            file_dir = '/media/zhujun/0DFD06D20DFD06D2/MVS/FrameMVS/data/1115-strip20-no-online/r3live_output/data_for_mesh_thu/rgb_pt.ply'
            all_data = o3d.io.read_point_cloud(file_dir)
            all_pts = np.asarray(all_data.points)
            all_cls = np.asarray(all_data.colors)
            all_pts_sel = all_pts[all_pts[:,2]<1.5]
            all_cls_sel = all_cls[all_pts[:,2]<1.5]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(all_pts)
            pcd.colors = o3d.utility.Vector3dVector(all_cls)
            inliers_ids_torch = simpletools.ransac_plane(torch.from_numpy(all_pts).float(), 0.02, 10, 1000) #* 13.0382s 1.71375s(cuda)  7.77573s
            inliers_ids = [v for v in inliers_ids_torch.cpu().numpy()]
            inlier_cloud = pcd.select_by_index(inliers_ids)
            inlier_cloud.paint_uniform_color([1.0, 0, 0])
            outlier_cloud = pcd.select_by_index(inliers_ids, invert=True)
            import pdb;pdb.set_trace()
            o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
            o3d.io.write_point_cloud("output/ply/ground_plane.ply", inlier_cloud + outlier_cloud)
            o3d.io.write_point_cloud("output/ply/inlier_cloud.ply", inlier_cloud)

            inliers_ids = [v for v in inliers_ids_torch.cpu().numpy()]
            start_time = time.time()
            plane_model, inliers = pcd.segment_plane(distance_threshold=0.02,ransac_n=10,num_iterations=1000)
            cal_time = time.time() - start_time
            print(cal_time) #* 28.580880165100098
            inlier_cloud = pcd.select_by_index(inliers_ids)
            inlier_cloud.paint_uniform_color([1.0, 0, 0])
            outlier_cloud = pcd.select_by_index(inliers_ids, invert=True)
            o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
            o3d.io.write_point_cloud("output/ply/ground_plane.ply", inlier_cloud + outlier_cloud)
            o3d.io.write_point_cloud("output/ply/inlier_cloud.ply", inlier_cloud)
            inlier_cloud = pcd.select_by_index(inliers)
            inlier_cloud.paint_uniform_color([1.0, 0, 0])
            outlier_cloud = pcd.select_by_index(inliers, invert=True)
            o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
            # o3d.io.write_point_cloud("output/ply/ground_plane.ply", inlier_cloud + outlier_cloud)
            # o3d.io.write_point_cloud("output/ply/inlier_cloud.ply", inlier_cloud)

            import pdb;pdb.set_trace()
            # import pdb;pdb.set_trace()
            o3d_tsdf_fusion_single = TSDFFusion(voxel_size=0.02)
            o3d_tsdf_fusion_single.integrate(depth_test, rgb, pose, K)
            mesh_o3d = o3d_tsdf_fusion_single.marching_cube(path=os.path.join('output', 'ply', f'o3d_single_New_{iter_num}.ply'))
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(sparse_pts[:,:3])
            pcd.colors = o3d.utility.Vector3dVector(sparse_pts[:,3:6]/255.0)
            start_time = time.time()
            plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,ransac_n=10,num_iterations=1000)
            inliers = inliers_ids
            cal_time = time.time() - start_time
            print(cal_time) #* 5.9947669506073
            # import pdb;pdb.set_trace()
            [a, b, c, d] = plane_model
            print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
            inlier_cloud = pcd.select_by_index(inliers)
            inlier_cloud.paint_uniform_color([1.0, 0, 0])
            outlier_cloud = pcd.select_by_index(inliers, invert=True)
            # outlier_cloud.paint_uniform_color([0, 1, 0])
            o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
            o3d.io.write_point_cloud("output/ply/segment_plane.ply", inlier_cloud + outlier_cloud)
            import pdb;pdb.set_trace()
            plt_imgs(images_plot,show=True,save=False)
            
            normal_test = np.random.rand(height, width, 3)
            normal_test = result[:,:,:3].cpu().numpy()
            # normal_test_norm = np.linalg.norm(normal_test,axis=2)
            # valid_normal_test = normal_test_norm>0
            # normal_test[~valid_normal_test,:] = [1,0,0]
            # normal_test = normal_test / np.linalg.norm(normal_test,axis=2)[:,:,None]
            # sparse_pts = depth2ply(depth_new, image=images_tensor[0].permute(1,2,0), K=intrinsics[0,0], T=extrinsics[0][0],return_pts = False, normal = normal_test)
            
            #* 拟合测试
            u, v = np.meshgrid(np.arange(0, width),np.arange(0, height))
            # import pdb;pdb.set_trace()
            uv1 = np.stack([u.reshape(-1),v.reshape(-1)],axis=0) #* (2, 1310720)
            depth_list = depth_np.reshape(-1)
            valid_mask = (~invalid_mask).reshape(-1)
            sel_uv = uv1[:,valid_mask] #* (2, 369380)
            sel_depth = depth_list[valid_mask]
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(sel_uv[0,::5], sel_uv[1,::5], sel_depth[::5], c='r', label='顺序点')
            plt.show()

            import pdb;pdb.set_trace()
            mesh = trimesh.Trimesh(vertices=np.asarray(mesh_o3d.vertices),faces=np.asarray(mesh_o3d.triangles),vertex_normals=np.asarray(mesh_o3d.vertex_normals))
            o3d.visualization.draw_geometries([mesh],point_show_normal=True)
            # test_depth = simpletools.depth_interp_comp(init_depth.squeeze().float(), edge_maps_tensor[0].squeeze().float())
            # #? 位姿优化
            # edge_maps_tensor_invalid = edge_maps_tensor<0.01
            # edge_maps_tensor_invalid = edge_maps_tensor_invalid.repeat(1,2,1,1)
            # _, edge_maps_squared_grad_tensor, edge_maps_grad_xy_tensor = cal_gray_grad(edge_maps_tensor)
            # # #* edge_maps_grad_xy_tensor.shape torch.Size([6, 2, 1024, 1280])
            # # #* edge_maps_squared_grad_tensor.shape torch.Size([6, 1, 1024, 1280])
            # _, edge_maps_squared_gray_grad, edge_maps_gray_grad_xy = cal_wider_gray_grad(edge_maps_tensor,offset = None, win_size = 9,ths=0.1, step=1)
            # edge_maps_grad_xy_tensor[edge_maps_tensor_invalid] = edge_maps_gray_grad_xy[edge_maps_tensor_invalid]
            # # edge_maps_squared_grad_tensor[edge_maps_tensor_invalid[:,0:1,:,:]] = edge_maps_squared_gray_grad[edge_maps_tensor_invalid[:,0:1,:,:]]
            # edge_feats = torch.cat([edge_maps_tensor,edge_maps_grad_xy_tensor],dim=1) #* torch.Size([6, 3, 1024, 1280])
            #? 深度图转为三维点
            # sparse_pts = depth2ply(init_depth, image=images_tensor[0].permute(1,2,0), K=intrinsics[0,0], T=extrinsics[0][0],return_pts = False)
            #? 生成点云测试
            depth_np = init_depth.squeeze().float().cpu().numpy()
            rgb = images_tensor[0].permute(1,2,0).cpu().numpy()
            pose = np.linalg.inv(extrinsics[0][0].cpu().numpy())
            K = intrinsics[0,0].cpu().numpy()
            self.o3d_tsdf_fusion.integrate(depth_np, rgb, pose, K)
            o3d_tsdf_fusion_single = TSDFFusion(voxel_size=0.02)
            o3d_tsdf_fusion_single.integrate(depth_np, rgb, pose, K)
            mesh_o3d = o3d_tsdf_fusion_single.marching_cube(path=os.path.join('output', 'ply', f'o3d_single_{iter_num}.ply'))
            
            def plot_render_test():
                height, width = depth_np.shape
                render = o3d.visualization.rendering.OffscreenRenderer(width, height)
                mat = o3d.visualization.rendering.MaterialRecord()
                mat.shader = 'defaultLit'
                render.scene.add_geometry("sphere1", mesh_o3d, mat)
                ext = extrinsics[0][0].cpu().numpy()
                height, width = depth_np.shape
                render.setup_camera(K, ext, width, height)
                cimg = np.asarray(render.render_to_image())
                dimg = np.asarray(render.render_to_depth_image())
                z_near = render.scene.camera.get_near()
                z_far = render.scene.camera.get_far()
                depth_render = 2.0 * z_near * z_far / (z_far + z_near - (2.0 * dimg - 1.0) * (z_far - z_near))
                depth_render[depth_render==depth_render.max()] = 0.0
                print(depth_render.min(),depth_render.max())
                images_plot=[]
                images_plot.append(['cimg', cimg/255.0,'jet'])
                images_plot.append(['dimg', depth_render,'jet'])
                images_plot.append(['rgb', rgb/255,'jet'])
                images_plot.append(['depth_np', depth_np,'jet'])
                diff = cimg - rgb
                mse = np.mean(np.square(diff))
                psnr = 10 * np.log10(255 * 255 / mse)
                plt_imgs(images_plot,show=True,save=False)
                psnr = 0.0
                psnr = compare_psnr(cimg , rgb.astype(np.uint8))
                ssim = compare_ssim(cimg , rgb.astype(np.uint8), multichannel=True)
            # plot_render_test()
            # import pdb;pdb.set_trace()
            # import pdb;pdb.set_trace()
            # pts2mesh(sparse_pts)
            #? 点云下采样,借助open3d下采样
            # lidar_pcd = o3d.geometry.PointCloud()
            # lidar_pcd.points = o3d.utility.Vector3dVector(sparse_pts[:,:3])
            # lidar_pcd.colors = o3d.utility.Vector3dVector(sparse_pts[:,3:6]/255.0)
            # o3d.visualization.draw_geometries([mvs_pts_ds_used_for_T_pcd])
            # draw_registration_result(mvs_pts_ds_used_for_T_pcd, lidar_pcd_ds, new_T)
            # o3d.io.write_point_cloud("output/ply/test_comb.ply", mvs_pcd.transform(trans_init)+lidar_pcd)
            # o3d.io.write_point_cloud("output/ply/mvs_pcd.ply", mvs_pcd)
            # o3d.io.write_point_cloud("output/ply/lidar_pcd.ply", lidar_pcd)
            # o3d.io.write_point_cloud("output/ply/test_comb.ply", mvs_pcd.transform(trans_init)+lidar_pcd)
            # Global_map.append_points_to_global_map(torch.from_numpy(sparse_pts).clone().float())
            # ori_ext = extrinsics[0][0]*1.0
            # new_depth = self.sparse_volume.proj2depth(ori_ext, self.local_mvs_dataset.K_torch.type_as(ori_ext), images_tensor.shape[-1], images_tensor.shape[-2])
            # new_sparse_pts = depth2ply(new_depth, image=images_tensor[0].permute(1,2,0)[:,:,[2,1,0]], K=intrinsics[0,0], T=ori_ext,return_pts = True)
            # new_lidar_pcd = o3d.geometry.PointCloud()
            # new_lidar_pcd.points = o3d.utility.Vector3dVector(new_sparse_pts[:,:3])
            # new_lidar_pcd.colors = o3d.utility.Vector3dVector(new_sparse_pts[:,3:6]/255.0)
            # o3d.io.write_point_cloud("output/ply/new_lidar_pcd.ply", new_lidar_pcd)
            import pdb;pdb.set_trace()
            print("cost time per iter: ", time.time() - start, "s")
            start = time.time()
        self.final_mesh = self.o3d_tsdf_fusion.marching_cube(path=os.path.join('output', 'ply', 'o3d_final.ply'))
        # import pdb;pdb.set_trace()

    def evaluation(self):
        #? 获取当前局部MVS数据,cuda
        self.local_mvs_dataset.clear_data()
        sample_data = self.local_mvs_dataset.get_next_data_all()
        images_tensor = sample_data['images'] #* torch.Size([3, 3, 1024, 1280])
        rgb = images_tensor[0].permute(1,2,0).cpu().numpy()
        height, width = rgb.shape[:2]
        self.local_mvs_dataset.clear_data()
        render = o3d.visualization.rendering.OffscreenRenderer(width, height)
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = 'defaultLit'
        self.final_mesh = o3d.io.read_triangle_mesh("/media/zhujun/0DFD06D20DFD06D2/MVS/FrameMVS/output/ply/o3d_final.ply")
        render.scene.add_geometry("final_mesh", self.final_mesh, mat)
        iter_num = 0
        start = time.time()
        stop = False
        psnr_list =[]
        ssim_list = []
        index_list = []
        reproj_list = []
        reproj_median_list = []
        last_ext = None
        last_rgb = None
        last_depth = None
        valid_all = []
        num_all = 0
        while(~stop):
            print("iter_num: ",iter_num)
            sample_data = self.local_mvs_dataset.get_next_data_all()
            if sample_data is None:
                break
            iter_num+=1
            indices = sample_data['indices'].cpu().numpy().astype(int) #* [ref_idx, src_idx1, ...]
            images_tensor = sample_data['images'] #* torch.Size([3, 3, 1024, 1280])
            intrinsics = sample_data['ints'] #* torch.Size([1, 3, 3, 3])
            extrinsics = sample_data['exts'] #* torch.Size([1, 3, 4, 4])
            rgb = images_tensor[0].permute(1,2,0).cpu().numpy()
            K = intrinsics[0,0].cpu().numpy()
            ext = extrinsics[0][0].cpu().numpy()
            render.setup_camera(K, ext, width, height)
            cimg = np.asarray(render.render_to_image())
            dimg = np.asarray(render.render_to_depth_image())
            z_near = render.scene.camera.get_near()
            z_far = render.scene.camera.get_far()
            depth_render = 2.0 * z_near * z_far / (z_far + z_near - (2.0 * dimg - 1.0) * (z_far - z_near))
            depth_render[depth_render==depth_render.max()] = 0.0
            # cimg[depth_render==0] = rgb.astype(np.uint8)[depth_render==0]
            # print(depth_render.min(),depth_render.max())
            if indices[0]==40 or 0:
                images_plot=[]
                images_plot.append(['cimg', cimg/255.0,'jet'])
                images_plot.append(['dimg', depth_render,'jet'])
                images_plot.append(['rgb', rgb/255,'jet'])
                images_plot.append(['depth_np', depth_render,'jet'])
                plt_imgs(images_plot,show=True,save=False)
                cv2.imwrite("tmp/cimg.png",cimg[:,:,[2,1,0]])
                cv2.imwrite("tmp/rgb.png",rgb.astype(np.uint8)[:,:,[2,1,0]])
                import pdb;pdb.set_trace()
            # (cimg.astype(np.float32) - rgb)**2
            # valid_depth = depth_render>0
            # valid_squared_error = np.sum((cimg.astype(np.float32) - rgb)**2)/np.sum(valid_depth)
            # 10 * np.log10((255 ** 2) / valid_squared_error)
            psnr = compare_psnr(cimg , rgb.astype(np.uint8))
            ssim = compare_ssim(cimg , rgb.astype(np.uint8), multichannel=True)
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            index_list.append(indices[0])
            # plot_render_test()
            # import pdb;pdb.set_trace()
            #* 计算重投影误差
            if last_ext is not None:
                width, height = depth_render.shape[1], depth_render.shape[0]
                x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
                depth_reprojected, x_reprojected, y_reprojected, valid_mask = reproject_with_depth(depth_render,K,ext,last_depth,K,last_ext)
                # check |p_reproject - p_1| < 1
                dist = np.sqrt((x_reprojected - x_ref) ** 2 + (y_reprojected - y_ref) ** 2)
                dist[~valid_mask]=0.0
                # check |d_reproject - d_1| / d_1 < 0.01
                depth_diff = np.abs(depth_reprojected - depth_render)
                relative_depth_diff = depth_diff / depth_render
                def test_plot():
                    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
                    surf = ax.plot_surface(x_ref, y_ref, dist,  cmap='rainbow')
                    # plot(cimg)
                    # plot(last_rgb)
                    # plot(depth_render)
                    # plot(last_depth)
                    images_plot=[]
                    images_plot.append(['cimg', cimg/255.0,'jet'])
                    images_plot.append(['dimg', depth_render,'jet'])
                    images_plot.append(['last_rgb', last_rgb/255,'jet'])
                    images_plot.append(['last_depth', last_depth,'jet'])
                    images_plot.append(['dist', dist,'jet'])
                    plt_imgs(images_plot,show=False,save=False)
                    plt.show()
                reproj_list.append(dist.sum()/valid_mask.sum())
                valid_v = dist[valid_mask]
                for v in valid_v:
                    valid_all.append(v)
                num_all += valid_mask.sum()
                # import pdb;pdb.set_trace()
                reproj_median_list.append(np.median(valid_v))
                if indices[0]==40 or indices[0]==493 or 0:
                    import pdb;pdb.set_trace()
            last_ext = ext
            last_rgb = cimg
            last_depth = depth_render
            print("cost time per iter: ", time.time() - start, "s")
            start = time.time()
        # self.final_mesh = self.o3d_tsdf_fusion.marching_cube(path=os.path.join('output', 'ply', 'o3d_final.ply'))
        valid_all_np = np.asarray(valid_all).reshape(-1,1)
        np.median(valid_all_np)
        valid_all_np.sum() / num_all
        import pdb;pdb.set_trace()


        psnr_np = np.asarray(psnr_list).reshape(-1,1)
        ssim_np = np.asarray(ssim_list).reshape(-1,1)
        index_np = np.asarray(index_list).reshape(-1,1)
        reproj_np = np.asarray(reproj_list).reshape(-1,1)
        reproj_median = np.asarray(reproj_median_list).reshape(-1,1)
        result_np = np.concatenate([index_np,psnr_np,ssim_np],axis=1)
        np.savetxt(os.path.join('tmp','result.txt'),result_np)
        plt.figure(figsize=(6,5))
        plt.plot(result_np[:,0], result_np[:,1])
        plt.title("PSNR")
        plt.xlabel("frame index")
        plt.ylabel("PSNR value")
        plt.grid("on")
        plt.tight_layout()
        plt.savefig('tmp/PSNR.png',dpi=500) #! figsize=(10,10) resolution=(500*10,500*10)=(5k,5k)
        # plt.show()
        plt.figure(figsize=(6,5))
        plt.plot(result_np[:,0], result_np[:,2])
        plt.title("SSIM")
        plt.xlabel("frame index")
        plt.ylabel("SSIM value")
        plt.grid("on")
        plt.tight_layout()
        plt.savefig('tmp/SSIM.png',dpi=500) #! figsize=(10,10) resolution=(500*10,500*10)=(5k,5k)
        # plt.show()
        plt.figure(figsize=(6,5))
        plt.plot(result_np[1:,0], reproj_np)
        plt.title("REPROJ")
        plt.xlabel("frame index")
        plt.ylabel("REPROJ value")
        plt.grid("on")
        plt.tight_layout()
        plt.savefig('tmp/REPROJ.png',dpi=500) #! figsize=(10,10) resolution=(500*10,500*10)=(5k,5k)
        plt.figure(figsize=(6,5))
        plt.plot(result_np[1:,0], reproj_median)
        plt.title("REPROJ")
        plt.xlabel("frame index")
        plt.ylabel("REPROJ value")
        plt.grid("on")
        plt.tight_layout()
        plt.savefig('tmp/REPROJ_median.png',dpi=500) #! figsize=(10,10) resolution=(500*10,500*10)=(5k,5k)
        plt.show()
        import pdb;pdb.set_trace()

    def run(self):
        #? 获取当前局部MVS数据,cuda
        sample_data = self.local_mvs_dataset.get_next_data()
        indices = sample_data['indices'].cpu().numpy().astype(int) #* [ref_idx, src_idx1, ...]
        images_tensor = sample_data['images'] #* torch.Size([3, 3, 1024, 1280])
        intrinsics = sample_data['ints'] #* torch.Size([1, 3, 3, 3])
        extrinsics = sample_data['exts'] #* torch.Size([1, 3, 4, 4])
        orig_depth = sample_data['depth'] #* torch.Size([1, 1, 1024, 1280])
        #? 初步稀疏深度补全，局部线性插值，置信度较高
        sparse2comp_bi = sparse_depth_to_dense(orig_depth).unsqueeze(1)
        sample_data['sparse2comp_bi'] = sparse2comp_bi
        init_depth = sparse2comp_bi #* torch.Size([1, 1, 1024, 1280])
        #? 轮廓提取
        edge_maps_tensor = images_seg.get_edge_maps_tensor(images_tensor) #* torch.Size([3, 1, 1024, 1280])
        #? 基于轮廓的区域分割
        seg_masks_tensor = images_seg.get_region_seg_tensor(edge_maps_tensor)
        
        # import pdb;pdb.set_trace()
        #? 位姿优化
        edge_maps_tensor_invalid = edge_maps_tensor<0.01
        edge_maps_tensor_invalid = edge_maps_tensor_invalid.repeat(1,2,1,1)
        _, edge_maps_squared_grad_tensor, edge_maps_grad_xy_tensor = cal_gray_grad(edge_maps_tensor)
        # #* edge_maps_grad_xy_tensor.shape torch.Size([6, 2, 1024, 1280])
        # #* edge_maps_squared_grad_tensor.shape torch.Size([6, 1, 1024, 1280])
        _, edge_maps_squared_gray_grad, edge_maps_gray_grad_xy = cal_wider_gray_grad(edge_maps_tensor,offset = None, win_size = 9,ths=0.1, step=1)
        edge_maps_grad_xy_tensor[edge_maps_tensor_invalid] = edge_maps_gray_grad_xy[edge_maps_tensor_invalid]
        # edge_maps_squared_grad_tensor[edge_maps_tensor_invalid[:,0:1,:,:]] = edge_maps_squared_gray_grad[edge_maps_tensor_invalid[:,0:1,:,:]]
        edge_feats = torch.cat([edge_maps_tensor,edge_maps_grad_xy_tensor],dim=1) #* torch.Size([6, 3, 1024, 1280])
        # src_index = 0
        # proj = extrinsics.clone()
        # proj[:, :, :3, :4] = torch.matmul(intrinsics, extrinsics[:, :, :3, :4])
        # proj = torch.unbind(proj, 1)
        # ref_proj, src_proj = proj[0], proj[1:] #* 从世界坐标系到相机坐标系
        # # #* 将参考图投影到源图得到参考图中的每个像素在源图中对应的坐标处的值，即灰度和梯度
        # src2ref_edge_feats, _ = differentiable_warping(edge_feats[src_index+1:src_index+2],src_proj[src_index],ref_proj,init_depth,return_uv=True)
        # src2ref_edge_feats = src2ref_edge_feats.squeeze(2) #* torch.Size([1, 3, 1024, 1280])
        # add_image_hori, add_image_vert = plt_sec(edge_maps_tensor[0:1], src2ref_edge_feats[:,0:1,:,:], show=True)
        # extrinsics = optimize_pose(images_tensor,intrinsics, extrinsics,init_depth,edge_feats,)
        # proj = extrinsics.clone()
        # proj[:, :, :3, :4] = torch.matmul(intrinsics, extrinsics[:, :, :3, :4])
        # proj = torch.unbind(proj, 1)
        # ref_proj, src_proj = proj[0], proj[1:] #* 从世界坐标系到相机坐标系
        # # * 将参考图投影到源图得到参考图中的每个像素在源图中对应的坐标处的值，即灰度和梯度
        # src2ref_edge_feats, _ = differentiable_warping(edge_feats[src_index+1:src_index+2],src_proj[src_index],ref_proj,init_depth,return_uv=True)
        # src2ref_edge_feats = src2ref_edge_feats.squeeze(2) #* torch.Size([1, 3, 1024, 1280])
        # add_image_hori, add_image_vert = plt_sec(edge_maps_tensor[0:1], src2ref_edge_feats[:,0:1,:,:], show=True)
        # optimize_depth(images_tensor,intrinsics, extrinsics,init_depth,edge_feats,)
        # plot(sample_data['depth'])
        # plot(init_depth)
        # plt.show()
        #? 局部 MVS index
        self.mvs_indices.append(indices)
        ref_idx = indices[0]
        src_ides = indices[1:]
        #? 局部 MVS 深度估计
        ACMMP.acmmp_init_test(cfg.data_path, ref_idx, images_tensor, intrinsics[0,0], extrinsics[0], sparse2comp_bi.squeeze().float())
        depth_test = ACMMP.GetDepth() #* 深度
        cost_test = ACMMP.GetCosts() #* 代价
        # depth_edge = ACMMP.GetDepthEdge()
        # init_depth1 = init_depth.squeeze()*1.0
        # init_depth1[depth_edge==0]=0.0
        # plot(init_depth,show=True)
        # import pdb;pdb.set_trace()
        #? 无效点去除
        nan_mask = depth_test!=depth_test
        depth_test[nan_mask]=0.0
        nan_mask = cost_test!=cost_test
        cost_test[nan_mask]=cost_test.max()
        invalid_mask = (depth_test>self.mvs_range[1]) | (depth_test<self.mvs_range[0])
        depth_test[invalid_mask]=0.0
        invalid_mask = (sparse2comp_bi>self.lidar_range[1]) | (sparse2comp_bi<self.lidar_range[0])
        sparse2comp_bi[invalid_mask] = 0.0
        edge_map_np = edge_maps_tensor[0].squeeze().cpu().numpy()
        
        def plot_test(ths=0.5):
            seg_masks_tensor = images_seg.get_region_seg_tensor(edge_maps_tensor, ths) #* torch.Size([3, 1024, 1280])
            images_plot=[]
            images_plot.append(['images_tensor',images_tensor[0].permute(1,2,0).squeeze().cpu().numpy()/255.0,'jet'])
            images_plot.append(['edge_maps_tensor',edge_maps_tensor[0].squeeze().cpu().numpy(),'jet'])
            images_plot.append(['seg_masks_tensor',seg_masks_tensor[0].squeeze().cpu().numpy(),'jet'])
            images_plot.append(['depth_test',depth_test.squeeze().cpu().numpy(),'jet'])
            plt_imgs(images_plot,show=True)
        #? mvs代价测试，当前的代价不能有效滤除无效点
        # self.diff_cost_test(depth_test, cost_test, images_tensor, intrinsics, extrinsics)
        # plot(sparse2comp_bi,show=True)
        #? 深度图转为三维点
        acmmp_pts = depth2ply(depth_test, image=images_tensor[0].permute(1,2,0), K=intrinsics[0,0], T=extrinsics[0][0],return_pts = True)
        sparse_pts = depth2ply(sparse2comp_bi, image=images_tensor[0].permute(1,2,0), K=intrinsics[0,0], T=extrinsics[0][0],return_pts = True)
        #? 点云下采样,借助open3d下采样
        mvs_pcd = o3d.geometry.PointCloud()
        mvs_pcd.points = o3d.utility.Vector3dVector(acmmp_pts[:,:3])
        mvs_pcd.colors = o3d.utility.Vector3dVector(acmmp_pts[:,3:6]/255.0)
        # o3d.visualization.draw_geometries([mvs_pcd])
        lidar_pcd = o3d.geometry.PointCloud()
        lidar_pcd.points = o3d.utility.Vector3dVector(sparse_pts[:,:3])
        lidar_pcd.colors = o3d.utility.Vector3dVector(sparse_pts[:,3:6]/255.0)
        mvs_pcd_ds = o3d.geometry.PointCloud.voxel_down_sample(mvs_pcd, self.cfg.voxel_down_sample_size)
        lidar_pcd_ds = o3d.geometry.PointCloud.voxel_down_sample(lidar_pcd, self.cfg.voxel_down_sample_size)
        def test_mvs(ths=0.5,save=False):
            depth_test_tmp = depth_test*1.0
            depth_test_tmp[edge_map_np<ths] = 0.0
            acmmp_pts_tmp = depth2ply(depth_test_tmp, image=images_tensor[0].permute(1,2,0), K=intrinsics[0,0], T=extrinsics[0][0],return_pts = True)
            mvs_pcd_tmp = o3d.geometry.PointCloud()
            mvs_pcd_tmp.points = o3d.utility.Vector3dVector(acmmp_pts_tmp[:,:3])
            mvs_pcd_tmp.colors = o3d.utility.Vector3dVector(acmmp_pts_tmp[:,3:6]/255.0)
            o3d.visualization.draw_geometries([mvs_pcd_tmp])
            if save:
                o3d.io.write_point_cloud(f"output/ply/mvs_pcd_th_{int(ths*100)}.ply", mvs_pcd_tmp)
        import pdb;pdb.set_trace()
        # o3d.visualization.draw_geometries([lidar_pcd_ds])
        # o3d.visualization.draw_geometries([mvs_pcd_ds])
        mvs_pts_ds = torch.from_numpy(np.asarray(mvs_pcd_ds.points)) #* torch.Size([9018, 3])
        lidar_pts_ds = torch.from_numpy(np.asarray(lidar_pcd_ds.points)) #* lidar_pts_ds
        # o3d.visualization.draw_geometries([lidar_pcd])
        #? mvs 点云与 lidar点云对齐
        ACMMP.build_basic_tree(lidar_pts_ds)
        new_T = ACMMP.align_pts(mvs_pts_ds, torch.eye(4))
        trans_init = np.eye(4)
        #* 显示用于对齐的mvs点云
        used_mask =ACMMP.GetUsedMask()
        used_pts_used_for_T = mvs_pts_ds[used_mask>0,:]
        mvs_pts_ds_used_for_T_pcd = o3d.geometry.PointCloud()
        mvs_pts_ds_used_for_T_pcd.points = o3d.utility.Vector3dVector(used_pts_used_for_T[:,:3])
        bwr = mpl.colormaps['bwr']
        used_dist = np.linalg.norm(used_pts_used_for_T,axis=1)
        used_dist_norm = (used_dist-used_dist.min())/(used_dist.max()-used_dist.min())
        used_color = bwr(used_dist_norm)
        mvs_pts_ds_used_for_T_pcd.colors = o3d.utility.Vector3dVector(used_color[:,:3])
        # o3d.visualization.draw_geometries([mvs_pts_ds_used_for_T_pcd])
        # draw_registration_result(mvs_pts_ds_used_for_T_pcd, lidar_pcd_ds, new_T)
        # o3d.io.write_point_cloud("output/ply/test_comb.ply", mvs_pcd.transform(trans_init)+lidar_pcd)
        # o3d.io.write_point_cloud("output/ply/mvs_pcd.ply", mvs_pcd)
        # o3d.io.write_point_cloud("output/ply/lidar_pcd.ply", lidar_pcd)
        # o3d.io.write_point_cloud("output/ply/test_comb.ply", mvs_pcd.transform(trans_init)+lidar_pcd)
        ori_ext = extrinsics[0][0]*1.0
        new_ext = ori_ext @ torch.inverse(new_T).type_as(ori_ext)
        new_depth = self.sparse_volume.proj2depth(new_ext, self.local_mvs_dataset.K_torch.type_as(ori_ext), images_tensor.shape[-1], images_tensor.shape[-2])
        new_sparse_pts = depth2ply(new_depth, image=images_tensor[0].permute(1,2,0)[:,:,[2,1,0]], K=intrinsics[0,0], T=new_ext,return_pts = True)
        new_lidar_pcd = o3d.geometry.PointCloud()
        new_lidar_pcd.points = o3d.utility.Vector3dVector(new_sparse_pts[:,:3])
        new_lidar_pcd.colors = o3d.utility.Vector3dVector(new_sparse_pts[:,3:6]/255.0)
        o3d.io.write_point_cloud("output/ply/new_lidar_pcd.ply", new_lidar_pcd)
        import pdb;pdb.set_trace()

# import pdb;pdb.set_trace()
local_mvs = LocalMVS(cfg)
# local_mvs.run()
# local_mvs.fusion()
local_mvs.evaluation()
import pdb;pdb.set_trace()
def ext_cal():
    from scipy.spatial.transform import Rotation as Rot
    # #* 原始点云处理部分
    # livox1 = o3d.io.read_point_cloud("/media/zhujun/0DFD06D20DFD06D2/catkin_ws/src/livox_camera_calib/data/1106bag/lidar2lidar/livox.ply")
    # livox2 = o3d.io.read_point_cloud("/media/zhujun/0DFD06D20DFD06D2/catkin_ws/src/livox_camera_calib/data/1106bag/lidar2lidar/livox_xt.ply")
    # livox1.paint_uniform_color([0, 0, 1])
    # livox2.paint_uniform_color([1, 0, 0])
    # livox1_ds = o3d.geometry.PointCloud.voxel_down_sample(livox1, 0.01)
    # mvs_pts_ds = np.asarray(livox2.points)
    # # 统计滤波
    # num_neighbors = 20  # K邻域点的个数
    # std_ratio = 1.0  # 标准差乘数
    # # 执行统计滤波，返回滤波后的点云sor_pcd和对应的索引ind
    # sor_pcd, ind = livox1_ds.remove_statistical_outlier(num_neighbors, std_ratio)
    # sor_pcd.paint_uniform_color([0, 0, 1])
    # print("统计滤波后的点云：", sor_pcd)
    # sor_pcd.paint_uniform_color([0, 0, 1])
    # # 提取噪声点云
    # sor_noise_pcd = livox1_ds.select_by_index(ind, invert=True)
    # print("噪声点云：", sor_noise_pcd)
    # sor_noise_pcd.paint_uniform_color([1, 0, 0])
    # # 可视化滤波结果
    # livox2_ds = o3d.geometry.PointCloud.voxel_down_sample(livox2, 0.01)
    # sor_pcd2, ind = livox2_ds.remove_statistical_outlier(num_neighbors, std_ratio)
    # o3d.visualization.draw_geometries([sor_pcd, sor_pcd2])
    # o3d.io.write_point_cloud("output/ply/livox2.ply", sor_pcd2)
    # o3d.io.write_point_cloud("output/ply/livox1.ply", sor_pcd)
    # pandar = o3d.io.read_point_cloud("/media/zhujun/0DFD06D20DFD06D2/catkin_ws/src/livox_camera_calib/data/1106bag/lidar2lidar/livox_xt_xt .ply")
    # livox2_ds = o3d.geometry.PointCloud.voxel_down_sample(pandar, 0.01)
    # livox2_ds.paint_uniform_color([0,1,0])
    # o3d.io.write_point_cloud("output/ply/pandar.ply", livox2_ds)
    #* livox 对齐
    # livox1 = o3d.io.read_point_cloud("output/ply/livox1.ply")
    # livox2 = o3d.io.read_point_cloud("output/ply/livox2.ply")
    
    # # o3d.io.write_point_cloud("output/ply/livox12_init.ply", livox1+livox2)
    # o3d.visualization.draw_geometries([livox1,livox2])
    # trans_init = np.eye(4)
    # trans_init[0,3] = 2
    # def test_angle(angle=31):
    #     r4 = Rot.from_euler('zxy', [angle,  0,  0], degrees=True).as_matrix()
    #     trans_init[:3,:3] = r4
    #     livox2_new = o3d.geometry.PointCloud()
    #     livox2_new += livox2
    #     livox2_new.transform(trans_init)
    #     o3d.visualization.draw_geometries([livox1,livox2_new])
    # r4 = Rot.from_euler('zxy', [31,  0,  0], degrees=True).as_matrix()
    # trans_init[:3,:3] = r4
    # livox2_new = o3d.geometry.PointCloud()
    # livox2_new += livox2
    # livox2_new.transform(trans_init)
    # livox1_pts = torch.from_numpy(np.asarray(livox1.points)) #* torch.Size([9018, 3])
    # livox2_pts = torch.from_numpy(np.asarray(livox2_new.points)) #* lidar_pts_ds
    # ACMMP.build_basic_tree(livox1_pts)
    # new_T = ACMMP.align_pts(livox2_pts, torch.eye(4))
    # livox2_new.transform(new_T)
    # o3d.visualization.draw_geometries([livox1,livox2_new])
    # for i in range(200):
    #     print(i)
    #     livox2_pts = torch.from_numpy(np.asarray(livox2_new.points))
    #     dist = torch.norm(livox2_pts,dim=1)
    #     livox2_pts = livox2_pts[dist>10]
    #     new_T = ACMMP.align_pts(livox2_pts, torch.eye(4))
    #     livox2_new.transform(new_T)
    #     if abs(new_T[:3,3]).max()<0.00001:
    #         o3d.visualization.draw_geometries([livox1,livox2_new])
    #         o3d.io.write_point_cloud("output/ply/livox12_opti.ply", livox1+livox2_new)
    #         import pdb;pdb.set_trace()
    #* 双雷达标定
    pandar = o3d.io.read_point_cloud("output/ply/pandar.ply")
    livox12 = o3d.io.read_point_cloud("output/ply/livox12_opti.ply")
    trans_init = np.eye(4)
    # trans_init[0,3] = 0
    def test_angle(ch=False,anglex=-90,angley=-90,anglez=0):
        trans_init = np.eye(4)
        r4 = Rot.from_euler('zxy', [anglez,  anglex, angley], degrees=True).as_matrix()
        trans_init[:3,:3] = r4
        pandar_new = o3d.geometry.PointCloud()
        pandar_new += pandar
        pandar_new.transform(trans_init)
        pandar_new.paint_uniform_color([1,1,0])
        if ch:
            pandar.transform(trans_init)
        else:
            o3d.visualization.draw_geometries([livox12,pandar_new,pandar])
    # o3d.visualization.draw_geometries([pandar,livox12])
    # o3d.io.write_point_cloud("output/ply/livox12_opti_pandar.ply", pandar+livox12)
    test_angle(True,-90,0,0)
    test_angle(True,0,0,-90)
    trans_init = np.eye(4)
    o3d.visualization.draw_geometries([pandar,livox12])
    def test_pos(ch=False,posx=0,posy=0,posz=0):
        trans_init = np.eye(4)
        # r4 = Rot.from_euler('zxy', [anglez,  anglex, angley], degrees=True).as_matrix()
        trans_init[0,3] = posx
        trans_init[1,3] = posy
        trans_init[2,3] = posz
        pandar_new = o3d.geometry.PointCloud()
        pandar_new += pandar
        pandar_new.transform(trans_init)
        pandar_new.paint_uniform_color([1,1,0])
        if ch:
            pandar.transform(trans_init)
        else:
            o3d.visualization.draw_geometries([livox12,pandar_new,pandar])
    test_pos(True,3,0.5,0)
    test_angle(True,0,0,30)
    test_pos(True,-0.4,-1.8,0)
    test_pos(False,0,0,0)
    livox12_pts = torch.from_numpy(np.asarray(livox12.points)) #* torch.Size([9018, 3])
    pandar_pts = torch.from_numpy(np.asarray(pandar.points)) #* lidar_pts_ds
    ACMMP.build_basic_tree(livox12_pts)
    new_T = ACMMP.align_pts(pandar_pts, torch.eye(4))
    pandar.transform(new_T)
    o3d.visualization.draw_geometries([pandar,livox12])
    import pdb;pdb.set_trace()
    for i in range(200):
        print(i)
        pandar_pts = torch.from_numpy(np.asarray(pandar.points))
        dist = torch.norm(pandar_pts,dim=1)
        # pandar_pts = pandar_pts[dist>10]
        new_T = ACMMP.align_pts(pandar_pts, torch.eye(4))
        pandar.transform(new_T)
        o3d.visualization.draw_geometries([pandar,livox12])
        if abs(new_T[:3,3]).max()<0.00001:
            o3d.visualization.draw_geometries([pandar,livox12])
            o3d.io.write_point_cloud("output/ply/livox12_opti_pandar_opti.ply", pandar+livox12)
            import pdb;pdb.set_trace()

    import pdb;pdb.set_trace()
# ext_cal()
# import pdb;pdb.set_trace()
exit(1)
#! 前期测试
local_mvs_dataset = LocalMVSDataset(cfg)
sample_data = local_mvs_dataset.get_next_data()
#* 稀疏深度插值初步补全，因为稀疏深度仅仅考虑较小的区域，因此深度置信度很高！
sparse2comp_bi = sparse_depth_to_dense(sample_data['depth']).unsqueeze(1)
sample_data['sparse2comp_bi'] = sparse2comp_bi
# informative_points_detection(sample_data)
images_tensor = sample_data['images'] #* torch.Size([3, 3, 1024, 1280])
intrinsics = sample_data['ints'] #* torch.Size([1, 3, 3, 3])
extrinsics = sample_data['exts'] #* torch.Size([1, 3, 4, 4])
init_depth = sample_data['sparse2comp_bi'] #* torch.Size([1, 1, 1024, 1280])
edge_maps_tensor = images_seg.get_edge_maps_tensor(images_tensor) #* torch.Size([3, 1, 1024, 1280])
seg_masks_tensor = images_seg.get_region_seg_tensor(edge_maps_tensor) #* torch.Size([3, 1024, 1280])
# images_seg.images_seg(images_tensor,show=True) 
# acmmp.tensorToMat(images_tensor)
# acmmp.acmmp_init(images_tensor, intrinsics[0,0], extrinsics[0])
# Params.test()
# print(intrinsics[0,0])
# print(extrinsics[0,0])
ref_idx = 0
ACMMP.acmmp_init_test(cfg.data_path, ref_idx, images_tensor, intrinsics[0,0], extrinsics[0], sparse2comp_bi.squeeze().float())
depth_test = ACMMP.GetDepth()
cost_test = ACMMP.GetCosts()
nan_mask = depth_test!=depth_test
depth_test[nan_mask]=0.0
nan_mask = cost_test!=cost_test
cost_test[nan_mask]=2.0
invalid_mask = (depth_test>40) | (depth_test<0.5) | (cost_test>0.1)
depth_test[invalid_mask]=0.0
invalid_mask = (sparse2comp_bi>40) | (sparse2comp_bi<2.5)
sparse2comp_bi[invalid_mask] = 0.0
# plot(sparse2comp_bi,show=True)
acmmp_pts = depth2ply(depth_test, image=images_tensor[0].permute(1,2,0)[:,:,[2,1,0]], K=intrinsics[0,0], T=extrinsics[0][0],return_pts = True)
sparse_pts = depth2ply(sparse2comp_bi, image=images_tensor[0].permute(1,2,0)[:,:,[2,1,0]], K=intrinsics[0,0], T=extrinsics[0][0],return_pts = True)
local_mvs_dataset.integrate_pts(sparse_pts)
sample_data_test = local_mvs_dataset.get_cur_data()
import pdb;pdb.set_trace()
sample_data_test['depth']
pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(acmmp_pts[:,:3])
pcd1.colors = o3d.utility.Vector3dVector(acmmp_pts[:,3:6]/255.0)
o3d.visualization.draw_geometries([pcd1])
pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(sparse_pts[:,:3])
pcd2.colors = o3d.utility.Vector3dVector(sparse_pts[:,3:6]/255.0)
pcd1_new = o3d.geometry.PointCloud.voxel_down_sample(pcd1, 0.05)
pcd2_new = o3d.geometry.PointCloud.voxel_down_sample(pcd2, 0.05)
import pdb;pdb.set_trace()
# o3d.visualization.draw_geometries([pcd2_new])
# o3d.visualization.draw_geometries([pcd1_new])
acmmp_pts_filtered = torch.from_numpy(np.asarray(pcd1_new.points)) #* torch.Size([9018, 3])
lidar_pts_ds = torch.from_numpy(np.asarray(pcd2_new.points)) #* lidar_pts_ds
# o3d.visualization.draw_geometries([pcd2])
ACMMP.build_basic_tree(lidar_pts_ds)
new_T = ACMMP.align_pts(acmmp_pts_filtered, torch.eye(4))
trans_init = np.eye(4)
used_mask =ACMMP.GetUsedMask()
used_pts = acmmp_pts_filtered[used_mask>0,:]
pcd1_used = o3d.geometry.PointCloud()
pcd1_used.points = o3d.utility.Vector3dVector(used_pts[:,:3])
bwr = mpl.colormaps['bwr']
used_dist = np.linalg.norm(used_pts,axis=1)
used_dist_norm = (used_dist-used_dist.min())/(used_dist.max()-used_dist.min())
used_color = bwr(used_dist_norm)
pcd1_used.colors = o3d.utility.Vector3dVector(used_color[:,:3])
import pdb;pdb.set_trace()
draw_registration_result(pcd1_used, pcd2_new, new_T)
o3d.io.write_point_cloud("output/ply/test_comb.ply", pcd1.transform(new_T)+pcd2)
# reg_p2p = o3d.pipelines.registration.registration_icp(pcd1, pcd2, 0.02, trans_init,
#         o3d.pipelines.registration.TransformationEstimationPointToPoint(),
#         # o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 2000),
#         )
# print(reg_p2p)
# print("Transformation is:")
# print(reg_p2p.transformation)
# draw_registration_result(pcd1, pcd2, reg_p2p.transformation)

import pdb;pdb.set_trace()

sparse_pts[:,3:6] = [255,0,0]
sparse_pts[:,0] += 0.1
pts = np.concatenate([acmmp_pts, sparse_pts],axis=0)
pcwrite("output/ply/depth2.ply",pts)
import pdb;pdb.set_trace()
# plot(seg_masks_tensor[0],show=True)

images_plot=[]
edge_maps_tensor_invalid = edge_maps_tensor<0.01
edge_maps_tensor_invalid = edge_maps_tensor_invalid.repeat(1,2,1,1)
_, edge_maps_squared_grad_tensor, edge_maps_grad_xy_tensor = cal_gray_grad(edge_maps_tensor)
#* edge_maps_grad_xy_tensor.shape torch.Size([6, 2, 1024, 1280])
#* edge_maps_squared_grad_tensor.shape torch.Size([6, 1, 1024, 1280])
_, edge_maps_squared_gray_grad, edge_maps_gray_grad_xy = cal_wider_gray_grad(edge_maps_tensor,offset = None, win_size = 9,ths=0.1, step=1)
edge_maps_grad_xy_tensor[edge_maps_tensor_invalid] = edge_maps_gray_grad_xy[edge_maps_tensor_invalid]
edge_maps_squared_grad_tensor[edge_maps_tensor_invalid[:,0:1,:,:]] = edge_maps_squared_gray_grad[edge_maps_tensor_invalid[:,0:1,:,:]]
edge_feats = torch.cat([edge_maps_tensor,edge_maps_grad_xy_tensor],dim=1) #* torch.Size([6, 3, 1024, 1280])
src_index = 0
proj = extrinsics.clone()
proj[:, :, :3, :4] = torch.matmul(intrinsics, extrinsics[:, :, :3, :4])
proj = torch.unbind(proj, 1)
ref_proj, src_proj = proj[0], proj[1:] #* 从世界坐标系到相机坐标系
#* 将参考图投影到源图得到参考图中的每个像素在源图中对应的坐标处的值，即灰度和梯度
src2ref_edge_feats, _ = differentiable_warping(edge_feats[src_index+1:src_index+2],src_proj[src_index],ref_proj,sparse2comp_bi,return_uv=True)
src2ref_edge_feats = src2ref_edge_feats.squeeze(2) #* torch.Size([1, 3, 1024, 1280])
add_image_hori, add_image_vert = plt_sec(edge_maps_tensor[0:1], src2ref_edge_feats[:,0:1,:,:])
extrinsics = optimize_pose(images_tensor,intrinsics, extrinsics,sparse2comp_bi,edge_feats,)
optimize_depth(images_tensor,intrinsics, extrinsics,sparse2comp_bi,edge_feats,)
plot(sample_data['depth'])
plot(sparse2comp_bi)
plt.show()
import pdb;pdb.set_trace()


#! 大量数据还是放在cpu，需要计算的放到GPU
# datapath = '/home/zhujun/catkin_ws/src/r3live-master/r3live_output/data_for_mesh'
# ply_dir = os.path.join(datapath, 'rgb_pt_small.ply')
# #* 深度图融合为ply
# # fuse_depth2ply(datapath,10)
# #* 深度图通过open3d融合为mesh
# # o3d_tsdf_fusion(datapath, num_depth=10, voxel_size=0.02, scalingFactor=1000.0)
# #* 读取ply数据保存到np中
# ply_np = read_ply(ply_dir) #* (8143478, 6) xyzrgb
# xyz_min = ply_np[:,:3].min()
# xyz_max = ply_np[:,:3].max()
# bound_max = 2*max(abs(xyz_min),abs(xyz_max))
# dimensions=np.asarray([bound_max, bound_max, bound_max])
# voxel_size=0.05
# n_feats=3
# #* 将np数据加入到hashmap中
# sparse_volume = SparseVolume(n_feats=n_feats, voxel_size=voxel_size, dimensions=dimensions, device="cpu:0")
# sparse_volume.integrate_pts(ply_np) #* sparse_volume.size() 6152961


#! 先不考虑生成mesh，反正可以通过点云生成mesh，也可以后面再投影到相机坐标系，在用tsdf融合。。。当前先主要考虑点云
#* from numpy to plt to mesh
# test_from_np2ply2mesh(datapath)


import pdb;pdb.set_trace()



