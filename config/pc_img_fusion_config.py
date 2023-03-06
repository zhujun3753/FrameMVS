import os

# data_path = '/media/zhujun/0DFD06D20DFD06D2/MVS/FrameMVS/data/1114/r3live_output/data_for_mesh_thu'
data_path = '/media/zhujun/0DFD06D20DFD06D2/MVS/FrameMVS/data/1115-strip20-no-online/r3live_output/data_for_mesh_thu'
ply_filename = 'rgb_pt.ply'
image_folder = "images"
depth_folder = "depth"
extrinsic_folder = "extrinsic"
intrinsic_folder = "intrinsic"
image_extension = ".png"
voxel_size=0.01
n_feats=3
num_views = 10
max_dim = -1
robust_train = False
sliding_win_len = 3
lidar_range = [2.5,40]
mvs_range = [0.5,40]
voxel_down_sample_size = 0.05
grid_res = 0.01
box_res = 0.1