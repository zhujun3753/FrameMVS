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
from config import pc_img_fusion_config as cfg
from utils import *
from models.module import differentiable_warping,proj_ref_src
from image_cpp_cuda_tool import images_seg
from ACMMP_net import *
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

#* 点云与图像融合，不考虑MVS的内容
class PointCloudImageFusion():
    def __init__(self,config):
        super(PointCloudImageFusion, self).__init__()
        self.cfg = config
        self.o3d_all_pts_ply = None
        # Global_map.str_test("asidhfbasdjhfbasjkdhfbasjkdhfb")
        #* 有效范围
        self.lidar_range = config.lidar_range
        self.mvs_range = config.mvs_range
        self.data_path = config.data_path
        self.max_dim = config.max_dim
        self.image_extension = config.image_extension
        image_list = [f for f in os.listdir(os.path.join(config.data_path, config.depth_folder)) if self.image_extension in f]
        image_list.sort(key=lambda f: int(f.split('.')[0]))
        self.image_filenames = [os.path.join(config.data_path, config.image_folder, fn)  for fn in image_list]
        #* 深度图是临时投影的
        # self.depth_filenames = [ os.path.join(config.data_path, config.depth_folder,fn)  for fn in image_list]
        self.ext_filenames = [ os.path.join(config.data_path, config.extrinsic_folder, fn.replace('.png','.txt'))  for fn in image_list]
        K_path =os.path.join(config.data_path, config.intrinsic_folder, 'intrinsic.txt')
        self.K = np.loadtxt(K_path)
        img = cv2.imread(self.image_filenames[0])
        height, width, _ = img.shape
        Global_map.set_K_wh(torch.from_numpy(self.K).float(), width, height)
        for i in range(len(self.image_filenames)):
            ext = np.loadtxt(self.ext_filenames[i])
            Global_map.add_img(torch.from_numpy(ext).float(), self.image_filenames[i])
            # print(Global_map.get_K())
        # import pdb;pdb.set_trace()
        near_ground_edge = o3d.io.read_point_cloud("output/ply/near_ground_edge.ply")
        near_ground_filter = o3d.io.read_point_cloud("output/ply/near_ground_filter.ply")
        ground_plane_param = np.loadtxt("output/ply/ground_plane_param.txt")
        near_ground_filter_pts = np.asarray(near_ground_filter.points)
        near_ground_edge_pts = np.asarray(near_ground_edge.points)
        ground_plane_param_torch = torch.from_numpy(ground_plane_param).float()
        near_ground_filter_pts_torch = torch.from_numpy(near_ground_filter_pts).float()
        near_ground_edge_pts_torch = torch.from_numpy(near_ground_edge_pts).float()
        filtered_ids_torch = Global_map.enrich_ground(near_ground_filter_pts_torch, near_ground_edge_pts_torch, ground_plane_param_torch)
        pcd = o3d.geometry.PointCloud()
        new_ground_pts = filtered_ids_torch.numpy()[:,:3]
        new_ground_pts[:,2] +=1
        pcd.points = o3d.utility.Vector3dVector(new_ground_pts)
        pcd.paint_uniform_color([0,1,0])
        # filtered_ids = [int(v) for v in filtered_ids_torch[:,0].cpu().numpy()]
        # o3d.io.write_point_cloud("output/ply/near_ground_color_edge_filter.ply", near_ground_filter+near_ground_edge.select_by_index(filtered_ids).paint_uniform_color([1.0, 0, 0]))
        # o3d.io.write_point_cloud("output/ply/near_ground_edge_filter.ply", near_ground_edge.select_by_index(filtered_ids))
        o3d.io.write_point_cloud("output/ply/new_ground.ply", pcd+near_ground_filter)


        import pdb;pdb.set_trace()
        # plot(filtered_ids_torch[:,:,0], True)
        # plot(filtered_ids_torch[:,:,0], True,'tmp/ground-proj.png')
        plot(filtered_ids_torch, True)
        # plot(filtered_ids_torch, True, 'tmp/ground-proj-comp.png')

        #* 每次都加载太慢了， 先处理完地面再说
        self.load_pc2hashmap()
        depth_test = Global_map.get_depth(0,False)
        img_test = Global_map.get_image(0)
        ext_test = Global_map.get_ext(0)
        K_test = Global_map.get_K()
        depth_test[depth_test>100]=0.0
        depth_np = depth_test.cpu().numpy()
        cur_pts_torch = Global_map.get_pc() #* torch.Size([13766780, 6])
        img_test1 = Global_map.ransac_fit_ground_plane(False, 0.01, 10, 1000)
        # import pdb;pdb.set_trace()
        inliers_ids_torch = Global_map.return_data("inlier_index_v")
        edge_ids_torch = Global_map.return_data("edge_index_v")
        inlier_filter_ids_torch = Global_map.return_data("inlier_index_v_filter")
        ground_plane_param =  Global_map.get_ground_plane_param()
        np.savetxt("output/ply/ground_plane_param.txt", ground_plane_param.numpy())

        # import pdb;pdb.set_trace()
        cur_pts_np = cur_pts_torch.cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cur_pts_np[:,:3])
        pcd.colors = o3d.utility.Vector3dVector(cur_pts_np[:,3:]/255.0)
        inliers_ids = [int(v) for v in inliers_ids_torch[:,0].cpu().numpy()]
        edge_ids = [int(v) for v in edge_ids_torch[:,0].cpu().numpy()]
        inlier_filter_ids = [int(v) for v in inlier_filter_ids_torch[:,0].cpu().numpy()]

        inlier_cloud = pcd.select_by_index(inliers_ids)
        inlier_cloud.paint_uniform_color([1.0, 0, 0])
        outlier_cloud = pcd.select_by_index(inliers_ids, invert=True)
        # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
        # o3d.io.write_point_cloud("output/ply/all_pts_with_near_ground.ply", inlier_cloud + outlier_cloud)
        o3d.io.write_point_cloud("output/ply/near_ground.ply", pcd.select_by_index(inliers_ids))
        o3d.io.write_point_cloud("output/ply/near_ground_edge.ply", pcd.select_by_index(edge_ids))
        o3d.io.write_point_cloud("output/ply/near_ground_filter.ply", pcd.select_by_index(inlier_filter_ids))
        o3d.io.write_point_cloud("output/ply/near_ground_color_edge.ply", pcd.select_by_index(inliers_ids)+pcd.select_by_index(edge_ids).paint_uniform_color([1.0, 0, 0]))
        import pdb;pdb.set_trace()
        #* 目前图像信息还是不太好用，图像所描述的区域比较局限，拟合的平面容易出现错开的情形
        #* 先考虑将地面点云补充一下
        edge_maps_tensor = images_seg.get_edge_maps_tensor(img_test.cuda()) #* torch.Size([3, 1, 1024, 1280])
        torch.cuda.empty_cache()
        # #? 基于轮廓的区域分割
        seg_masks_tensor = images_seg.get_region_seg_tensor(edge_maps_tensor, 0.01)
        torch.cuda.empty_cache()
        def plot_test(ths=0.5,save=False):
            seg_masks_tensor = images_seg.get_region_seg_tensor(edge_maps_tensor, ths) #* torch.Size([3, 1024, 1280])
            images_plot=[]
            images_plot.append(['images_tensor', img_test.squeeze().cpu().numpy()/255.0,'jet'])
            images_plot.append(['edge_maps_tensor', edge_maps_tensor[0].squeeze().cpu().numpy(),'jet'])
            segmask_np = shuffle_mask(seg_masks_tensor[0].squeeze().cpu().numpy())
            # segmask_np = seg_masks_tensor[0].squeeze().cpu().numpy()
            images_plot.append(['seg_masks_tensor', segmask_np,'jet'])
            images_plot.append(['init_depth', depth_test.squeeze().float().cpu().numpy(),'jet'])
            plt_imgs(images_plot,show=True,save=save)
        import pdb;pdb.set_trace()
        image_info_torch = torch.cat([
                img_test.permute(2,0,1).cpu().float(),  #* 3 rgb
                edge_maps_tensor[0].cpu().float(),  #* 1 edge
                seg_masks_tensor.cpu().float(), #* 1 seg
                ])
        results = Global_map.point_cloud_segmentation(image_info_torch,0,False)
        depth_new = results[:,:,0]
        depth_fit = results[:,:,1]
        img_test_np = img_test[:,:,[2,1,0]].numpy()
        img_test_np[depth_fit.numpy()>0] = [255,0,0]
        import pdb;pdb.set_trace()
        depth_new = Global_map.get_depth_with_attr(0,False)
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
        o3d.io.write_point_cloud("output/ply/all_pts_with_near_ground.ply", inlier_cloud + outlier_cloud)
        o3d.io.write_point_cloud("output/ply/near_ground.ply", pcd.select_by_index(inliers_ids))
        import pdb;pdb.set_trace()
        # depth2 = Global_map.bilinear_interplote_depth(depth_test)
        pts_new = depth2ply(depth_new, image=img_test[:,:,[2,1,0]].numpy(), K=K_test.numpy(), T=ext_test.numpy(),return_pts = True)
        pts_test = depth2ply(depth_test, image=img_test[:,:,[2,1,0]].numpy(), K=K_test.numpy(), T=ext_test.numpy(),return_pts = False)
        
        pts_fit = depth2ply(depth_fit, image=img_test_np, K=K_test.numpy(), T=ext_test.numpy(),return_pts = True)
        img_test_np[depth_fit.numpy()>0] = [0,255,0]
        pts_new = depth2ply(depth_new, image=img_test_np, K=K_test.numpy(), T=ext_test.numpy(),return_pts = True)
        pcwrite("output/ply/fit_new.ply",np.concatenate([pts_new, pts_fit], axis=0))

        # pts_test = depth2ply(depth2, image=img_test[:,:,[2,1,0]].numpy(), K=K_test.numpy(), T=ext_test.numpy(),return_pts = False)
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(cur_pts_np[inliers_ids,:3])
        # pcd.colors = o3d.utility.Vector3dVector(cur_pts_np[inliers_ids,3:]/255.0)
        # o3d.visualization.draw_geometries([pcd])
        import pdb;pdb.set_trace()
        o3d_tsdf_fusion_single = TSDFFusion(voxel_length=0.02, sdf_trunc=0.1)
        o3d_tsdf_fusion_single.integrate(depth_new.numpy(), img_test[:,:,[2,1,0]].numpy(), np.linalg.inv(ext_test.numpy()), K_test.numpy())
        mesh_o3d = o3d_tsdf_fusion_single.marching_cube(path=os.path.join('output', 'ply', f'o3d_single_{0}.ply'))
        import pdb;pdb.set_trace()
        plot(depth_new.numpy(),show=True)
        plot(depth_test.numpy(),show=True)
        plot(depth_new.numpy(),name='output/ply/smoothing-mesh/init_img_seg.png')
        plot(img_test[:,:,[2,1,0]].numpy(),show=True)
    
    def load_pc2hashmap(self,):
        #* 读取ply数据保存到np中
        file_dir = os.path.join(self.cfg.data_path, self.cfg.ply_filename)
        self.o3d_all_pts_ply = o3d.io.read_point_cloud(file_dir)
        Global_map.cur_resolution()
        Global_map.set_resolution(self.cfg.grid_res, self.cfg.box_res)
        #* 将np数据加入到hashmap中
        all_pts = np.asarray(self.o3d_all_pts_ply.points)
        all_cls = np.asarray(self.o3d_all_pts_ply.colors)
        ply_np = np.concatenate([all_pts, all_cls*255], axis=1)
        # Global_map.append_points_to_global_map(torch.from_numpy(ply_np[0:10000,:]).clone().float())
        Global_map.append_points_to_global_map(torch.from_numpy(ply_np).clone().float())

    def show_pc(self):
        if(self.o3d_all_pts_ply is not None):
            o3d.visualization.draw_geometries([self.o3d_all_pts_ply])

    def get_pts_cls(self):
        if(self.o3d_all_pts_ply is not None):
            all_pts = np.asarray(self.o3d_all_pts_ply.points)
            all_cls = np.asarray(self.o3d_all_pts_ply.colors)
            return [all_pts, all_cls]
        else:
            return [None, None]

    def simple_comp(self):
        #? 获取当前局部MVS数据,cuda
        iter_num = 0
        start = time.time()
        stop = False
        Global_map.set_resolution(0.01,0.1)
        self.o3d_tsdf_fusion = TSDFFusion(voxel_size=0.02)
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
            o3d_tsdf_fusion_single = TSDFFusion(voxel_size=0.02)
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

# import pdb;pdb.set_trace()
pc_img_fusion = PointCloudImageFusion(cfg)
pts_cls = pc_img_fusion.get_pts_cls()
all_pts = pts_cls[0]
all_cls = pts_cls[1]
pc_img_fusion.show_pc()
# local_mvs.run()
# local_mvs.simple_comp()
# local_mvs.evaluation()
import pdb;pdb.set_trace()