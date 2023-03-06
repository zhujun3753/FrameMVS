from curses import newwin
import json
import cv2
import os
from sqlalchemy import false
import torch
import trimesh
import numpy as np

import os
from plyfile import PlyData, PlyElement
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
# from third_parties.fusion import pcwrite
import open3d as o3d
from tqdm import tqdm
from tsdf_fusion.fusion import *
from utils import *

root_dir ='/media/zhujun/0DFD06D20DFD06D2/catkin_ws/src/r3live-orig/r3live_output/data_for_mesh_thu'
# root_dir ='/media/zhujun/0DFD06D20DFD06D2/MVS/FrameMVS/data/1114offline/r3live_output/data_for_mesh_thu'
root_dir ='/media/zhujun/0DFD06D20DFD06D2/MVS/FrameMVS/data/1114/r3live_output/data_for_mesh_thu'
root_dir = '/media/zhujun/0DFD06D20DFD06D2/MVS/FrameMVS/data/1114-no-tune2/r3live_output/data_for_mesh_thu'
root_dir = '/media/zhujun/0DFD06D20DFD06D2/MVS/FrameMVS/data/1115-strip20-no-online/r3live_output/data_for_mesh_thu'

def gen_depth():
    file_dir = root_dir+'/rgb_pt.pcd'  #文件的路径
    K_path = root_dir+'/intrinsic/intrinsic.txt'
    K = np.loadtxt(K_path)
    image_list = [f for f in os.listdir(root_dir+'/images') if 'png' in f]
    image_list.sort(key=lambda f: int(f.split('.')[0]))
    # import pdb;pdb.set_trace()
    # plydata = PlyData.read(file_dir)  # 读取文件
    # data = plydata.elements[0].data  # 读取数据 (1224539,)
    # data_pd = pd.DataFrame(data)  # 转换成DataFrame, 因为DataFrame可以解析结构化的数据 (1224539, 6)
    # data_np = np.zeros(data_pd.shape, dtype=np.float)  # 初始化储存数据的array
    # property_names = data[0].dtype.names  # 读取property的名字 ('x', 'y', 'z', 'red', 'green', 'blue')
    # for i, name in enumerate(property_names):  # 按property读取数据，这样可以保证读出的数据是同样的数据类型。
    #     data_np[:, i] = data_pd[name]
    data = o3d.io.read_point_cloud(file_dir)
    # o3d.visualization.draw_geometries([data])
    # import open3d as o3d
    

    import pdb;pdb.set_trace()
    if not os.path.exists(os.path.join(root_dir, 'rgb_pt.ply')):
        o3d.io.write_point_cloud(os.path.join(root_dir, 'rgb_pt.ply'), data)
        import pdb;pdb.set_trace()
    data_pts = torch.from_numpy(np.asarray(data.points)) #* torch.Size([9018, 3])
    data_crs = torch.from_numpy(np.asarray(data.colors)) #* torch.Size([9018, 3])
    data_np = np.concatenate([data_pts, data_crs], axis=1)
    # import pdb;pdb.set_trace()
    print(data_np.shape)
    step = 1
    # test_angle(anglex=0,angley=0,anglez=1.5):
    from scipy.spatial.transform import Rotation as Rot
    # trans_init = np.eye(4)
    # r4 = Rot.from_euler('zxy', [0, 0,  1.5], degrees=True).as_matrix()
    # trans_init[:3,:3] = r4
    few = True
    for idx in tqdm(range(0,len(image_list), step)):
        # print(idx, ' in ', len(image_list))
        if idx not in [0,177,562,752, 822,240,770,780,790] and few:
        # if idx not in [770,780,790] and few:
            continue
        ext_path = root_dir+'/extrinsic/'+image_list[idx].replace('.png','.txt')
        image_path = root_dir+'/images/'+image_list[idx]
        T_cw = np.loadtxt(ext_path)
        # from scipy.spatial.transform import Rotation as Rot
        trans_init = np.eye(4)
        r4 = Rot.from_euler('zxy', [0, -0.7,  -0.70], degrees=True).as_matrix()
        trans_init[:3,:3] = r4 #* 相机运动
        T_cw_new = T_cw*1.0
        T_wc_new = np.linalg.inv(T_cw_new)
        T_wc_new =  T_wc_new @ trans_init #* 相机运动 不能左乘
        #* 直接考虑相机位姿的运动，而不是坐标变换
        #* 坐标变换(左乘)改变的是坐标系，相机运动(右乘)改变的是相机位姿
        T_cw_new = np.linalg.inv(T_wc_new)
        # T_cw = T_cw_new #* 使用微调
        image = cv2.imread(image_path) # cv2默认为bgr顺序
        image = image[:,:,[2,1,0]]
        hight,width,_ = image.shape
        # plt.imshow(image)
        # plt.show()
        # import pdb;pdb.set_trace()
        data_np_indices = np.array([i for i in range(data_np.shape[0])])
        pts = data_np[:,:3]
        colors = data_np[:,3:]
        #* from world to cam
        pts_c_all = T_cw[:3,:3] @ pts.T + T_cw[:3,3:] #* (3, 1224539)
        valid_mask = pts_c_all[2,:]>0.2
        pts_c = pts_c_all[:,valid_mask] #* (3, 1146086)
        data_np_indices_c = data_np_indices[valid_mask]
        uvz = K @ pts_c
        z = uvz[2,:]
        uv = uvz[:2,:]/z #* (2, 1146086)
        uv_valid_mask = (uv[0,:]>0)*(uv[0,:]<width-1)*(uv[1,:]>0)*(uv[1,:]<hight-1)
        uv_valid = uv[:,uv_valid_mask].T #* (770288, 2)
        z_valid = z[uv_valid_mask]
        data_np_indices_c_v = data_np_indices_c[uv_valid_mask]
        depth_map=np.ones((hight,width))*np.inf
        data_np_indices_depth_map=np.ones((hight,width))*(-1)
        for uv_i in range(len(uv_valid)):
            u,v = int(uv_valid[uv_i,0]), int(uv_valid[uv_i,1])
            if depth_map[v,u]>z_valid[uv_i]:
                depth_map[v,u]=z_valid[uv_i]
                data_np_indices_depth_map[v,u]=data_np_indices_c_v[uv_i]
        data_np_indices_depth_map_flat = data_np_indices_depth_map.reshape(-1)
        data_np_indices_depth_map_flat=data_np_indices_depth_map_flat[data_np_indices_depth_map_flat>-1]
        depth_map[depth_map==np.inf] = 0
        file_name = root_dir+'/depth/'+ image_list[idx]
        cv2.imwrite(file_name,(depth_map*1000).astype(np.uint16))
        def test_angle(anglex=0,angley=0,anglez=1.5):
            from scipy.spatial.transform import Rotation as Rot
            trans_init = np.eye(4)
            r4 = Rot.from_euler('zxy', [anglez, anglex,  angley], degrees=True).as_matrix()
            trans_init[:3,:3] = r4 #* 相机运动
            T_cw_new = T_cw*1.0
            T_wc_new = np.linalg.inv(T_cw_new)
            T_wc_new =  T_wc_new @ trans_init #* 相机运动 不能左乘
            #* 直接考虑相机位姿的运动，而不是坐标变换
            #* 坐标变换(左乘)改变的是坐标系，相机运动(右乘)改变的是相机位姿
            T_cw_new = np.linalg.inv(T_wc_new)
            #* from world to cam
            pts_c_all = T_cw_new[:3,:3] @ pts.T + T_cw_new[:3,3:] #* (3, 1224539)
            valid_mask = pts_c_all[2,:]>0.2
            pts_c = pts_c_all[:,valid_mask] #* (3, 1146086)
            data_np_indices_c = data_np_indices[valid_mask]
            uvz = K @ pts_c
            z = uvz[2,:]
            uv = uvz[:2,:]/z #* (2, 1146086)
            uv_valid_mask = (uv[0,:]>0)*(uv[0,:]<width-1)*(uv[1,:]>0)*(uv[1,:]<hight-1)
            uv_valid = uv[:,uv_valid_mask].T #* (770288, 2)
            z_valid = z[uv_valid_mask]
            data_np_indices_c_v = data_np_indices_c[uv_valid_mask]
            depth_map=np.ones((hight,width))*np.inf
            data_np_indices_depth_map=np.ones((hight,width))*(-1)
            for uv_i in range(len(uv_valid)):
                u,v = int(uv_valid[uv_i,0]), int(uv_valid[uv_i,1])
                if depth_map[v,u]>z_valid[uv_i]:
                    depth_map[v,u]=z_valid[uv_i]
                    data_np_indices_depth_map[v,u]=data_np_indices_c_v[uv_i]
            data_np_indices_depth_map_flat = data_np_indices_depth_map.reshape(-1)
            data_np_indices_depth_map_flat=data_np_indices_depth_map_flat[data_np_indices_depth_map_flat>-1]
            depth_map[depth_map==np.inf] = 0
            sparse2comp_bi = sparse_depth_to_dense(torch.from_numpy(depth_map).cuda()[None,None,...]).unsqueeze(1)
            depth_np = sparse2comp_bi.squeeze().float().cpu().numpy()
            rgb = image
            pose = np.linalg.inv(T_cw)
            o3d_tsdf_fusion_single = TSDFFusion(voxel_size=0.02)
            o3d_tsdf_fusion_single.integrate(depth_np, rgb, pose, K)
            mesh_o3d = o3d_tsdf_fusion_single.marching_cube(path=os.path.join('output', 'ply', f'o3d_single_test{idx}.ply'))
        if idx%4==0 or few:
            # test_angle (0,-1.5,0)
            # test_angle (0,0,0)
            sparse2comp_bi = sparse_depth_to_dense(torch.from_numpy(depth_map).cuda()[None,None,...]).unsqueeze(1)
            depth_np = sparse2comp_bi.squeeze().float().cpu().numpy()
            rgb = image
            pose = np.linalg.inv(T_cw)
            o3d_tsdf_fusion_single = TSDFFusion(voxel_size=0.02)
            o3d_tsdf_fusion_single.integrate(depth_np, rgb, pose, K)
            mesh_o3d = o3d_tsdf_fusion_single.marching_cube(path=os.path.join('output', 'ply', f'o3d_single_test{idx}.ply'))
            # import pdb;pdb.set_trace()
        # if idx>100:
            # test_angle(0,0,1.5)
            # import pdb;pdb.set_trace()
        # print("Save to : ",file_name)
        # read_data = cv2.imread(root_dir+'/depth.png',-1)
        # plt.imshow(depth_map,cmap=plt.cm.autumn)
        # plt.show()
        # pcwrite(root_dir+"/r3live.ply",data_np[data_np_indices_c_v])
        # pcwrite(root_dir+"/r3live_depth.ply",data_np[data_np_indices_depth_map_flat.astype(int)])
    import pdb;pdb.set_trace()

def view_selection():
    image_list = [f for f in os.listdir(root_dir+'/depth') if 'png' in f]
    image_list.sort()
    depth_list = [ os.path.join(root_dir+'/depth',fn)  for fn in image_list]
    ext_list = [ os.path.join(root_dir+'/extrinsic',fn.replace('.png','.txt'))  for fn in image_list]
    K_path = root_dir+'/intrinsic/intrinsic.txt'
    K = np.loadtxt(K_path)
    K = torch.from_numpy(K).cuda()
    K_inv = torch.inverse(K)
    dtype=K.dtype
    device=K.device
    batch = 1
    T_cws = [torch.from_numpy(np.loadtxt(ext_list_i)).cuda() for ext_list_i in ext_list]
    T_wcs = [torch.inverse(T_cw_i) for T_cw_i in T_cws]
    score = np.zeros((len(image_list),len(image_list)))
    for depth_list_i in tqdm(range(len(depth_list))):
        depth_i = depth_list[depth_list_i]
        T_cw = T_cws[depth_list_i]
        T_wc = T_wcs[depth_list_i]
        depth = cv2.imread(depth_i,-1)/1000.0 # cv2默认为bgr顺序
        depth_torch = torch.from_numpy(depth).to(device)[None,None] #* torch.Size([1, 1, 1024, 1280])
        height,width = depth.shape
        win_size = 32
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
        win_xy_with_offset = torch.cat(win_xy_list, dim=2)  #* torch.Size([1, 2, 1024, 1280])
        win_x_normalized = win_xy_with_offset[:, 0, :, :] / ((width - 1) / 2) - 1
        win_y_normalized = win_xy_with_offset[:, 1, :, :] / ((height - 1) / 2) - 1
        win_grid = torch.stack((win_x_normalized, win_y_normalized), dim=3)  #* torch.Size([1, 1024, 1280, 2])
        # import pdb;pdb.set_trace()
        #* squared_gray_grad_patch.shape torch.Size([1, 1, 1024, 1280]) #* 总共1280个区域，每个区域共有1024个点，每个点一个平方灰度梯度
        depth_sample = F.grid_sample(depth_torch, win_grid, mode="nearest", padding_mode="reflection", align_corners=True)
        indices = depth_sample.max(dim=2).indices.squeeze().cpu().numpy()
        depths = depth_sample.max(dim=2).values.squeeze() #* torch.Size([1280])
        depth_uv = [win_xy_with_offset[0,:,indices[j],j] for j in range(1280)]
        depth_uv_torch = torch.stack(depth_uv,dim=0) #* torch.Size([1280, 2])
        depth_uv1_torch = torch.cat([depth_uv_torch,torch.ones_like(depth_uv_torch[:,0:1])],dim=1) #* torch.Size([1280, 3])
        depth_uvd_torch = depth_uv1_torch*depths.unsqueeze(1)
        depth_uvd_torch_valid = depth_uvd_torch[depths>0,:] #* torch.Size([1280, 3])
        depth_xyz_c = torch.matmul(K_inv,depth_uvd_torch_valid.T) #* torch.Size([3, 1280])
        depth_xyz1_w = torch.matmul(T_wc,torch.cat([depth_xyz_c,torch.ones_like(depth_xyz_c[0:1,:])],dim=0)) #* torch.Size([4, 1280])
        depth_xyz_w = depth_xyz1_w[:3,:]
        for j in range(depth_list_i+1, len(depth_list)):
            proj = T_cws[j]*1.0
            proj[:3,:] = torch.matmul(K,proj[:3,:])
            depth_uvd1_cj = torch.matmul(proj,depth_xyz1_w)
            depth_uv_cj = depth_uvd1_cj[:2]/(depth_uvd1_cj[2]+1e-6)
            valid_mask = (depth_uvd1_cj[2,:]>0.1)*(depth_uv_cj[0]>0)*(depth_uv_cj[0]<width)*(depth_uv_cj[1]>0)*(depth_uv_cj[1]<height) #* torch.Size([1280])
            selected_pts = depth_xyz_w[:,valid_mask] #* torch.Size([3, 772])
            #* vector to both camera center
            v1 = T_wcs[j][:3,3:4] - selected_pts
            v2 = T_wcs[depth_list_i][:3,3:4] - selected_pts #* torch.Size([3, 871])
            distance = (T_wcs[j][:3,3:4]-T_wcs[depth_list_i][:3,3:4]).norm() #* cm
            # distance = torch.clamp(distance,0,300)
            # import pdb;pdb.set_trace()
            cos_thetas = (v1*v2).sum(dim=0)/(v1.norm(dim=0)*v2.norm(dim=0)) #* torch.Size([871])
            thetas = torch.arccos(cos_thetas) #* (180 / torch.pi)*
            invalid_mask = torch.isnan(thetas)
            valid_theta = thetas[~invalid_mask]
            valid_theta = valid_theta[valid_theta<torch.pi/3]
            view_score = np.sqrt(np.sqrt(len(valid_theta)))*valid_theta.sum()/1000.0 * torch.exp(-torch.abs(distance-10)/10)
            score[depth_list_i,j] = view_score.cpu().numpy()
            score[j,depth_list_i] = view_score.cpu().numpy()
            if(depth_list_i==0):
                print(image_list[depth_list_i],", ", image_list[j], ", ", view_score.cpu().numpy(),', ',np.sqrt(np.sqrt(len(valid_theta))),", ",valid_theta.sum().cpu().numpy(),", ",distance.cpu().numpy(),', ',torch.exp(-torch.abs(distance-10)/10).cpu().numpy())
            # if(view_score.cpu().numpy()>4):
            #     import pdb;pdb.set_trace()
            # theta_all = (180 / np.pi) * np.arccos( np.sum(point2cam_i_v*point2cam_j_v,axis=1)/ np.linalg.norm(point2cam_i_v,axis=1) / np.linalg.norm(point2cam_j_v,axis=1))
            # theta_valid = np.array([theta for theta in theta_all if theta and not np.isnan(theta)]) #* 1112
            # # a2=0 if np.nan else 1
            # # (Pdb) a2
            # # 0  #* 离谱,
            # if len(theta_valid)<1:
            #     return ind1,ind2, 0
            # theta_weight = np.array([args.sigma1 if theta <= args.theta0 else args.sigma2 for theta in theta_valid ])
            # view_score = np.exp(-(theta_valid - args.theta0)**2 / (2 * theta_weight ** 2)) #* 1112
            # view_score_sum = np.sum(view_score)
            # return ind1,ind2, view_score_sum
    score_indices = np.argsort(score,axis=1)[:,::-1]
    with open(os.path.join(root_dir, "pair.txt"), "w") as f:
        f.write("%d\n" % len(score_indices))
        for i, sorted_score in enumerate(score_indices):
            f.write("%d\n%d " % (int(image_list[i].split('.')[0]), 10))
            for j in range(10):
                f.write("%d %f " % (int(image_list[score_indices[i,j]].split('.')[0]) , score[i,score_indices[i,j]]))
            f.write("\n")
    import pdb;pdb.set_trace()


if __name__ == "__main__":
    gen_depth()
    view_selection()


