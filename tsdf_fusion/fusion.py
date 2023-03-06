import numpy as np
from numba import njit, prange
from skimage import measure
import torch, pickle, os
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from PIL import Image
import cv2
from tqdm import tqdm
from tsdf_fusion.utils.o3d_helper import TSDFFusion
import open3d as o3d
from tsdf_fusion.sparse_volume import SparseVolume
from plyfile import PlyData
import pandas as pd
import matplotlib.pyplot as plt
import time

def pts2mesh(pts):
    shape = pts.shape
    if len(shape)!=2 or shape[1]<6:
        print("Wrong shape!")
        return None
    xyz = pts[:,:3]
    colors = pts[:,3:6]
    if colors.max()>10:
        colors = colors/255.0
    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # pcd.normals = o3d.utility.Vector3dVector(nxnynz)
    # o3d.io.write_point_cloud(datapath + "/sync.ply", pcd)

    # Load saved point cloud and visualize it
    o3d.visualization.draw_geometries([pcd])
    radius = 0.05   # 搜索半径
    max_nn = 30     # 邻域内用于估算法线的最大点数
    start = time.time()
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))     # 执行法线估计
    esti = time.time()
    print("esti - start: ", esti - start)
    radii = [0.05, 0.05, 0.05, 0.04]
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
    create_mesh = time.time()
    print("create_mesh - esti: ", create_mesh - esti)
    o3d.visualization.draw_geometries([pcd, rec_mesh])
    # o3d.io.write_triangle_mesh(datapath + "/sync_mesh.ply", rec_mesh)
    import pdb;pdb.set_trace()

def read_ply(ply_dir):
    print("Read ply from: ", ply_dir, " to numpy!")
    plydata = PlyData.read(ply_dir)  # 读取文件
    data = plydata.elements[0].data  # 读取数据 (1224539,)
    data_pd = pd.DataFrame(data)  # 转换成DataFrame, 因为DataFrame可以解析结构化的数据 (1224539, 6)
    data_np = np.zeros(data_pd.shape, dtype=np.float)  # 初始化储存数据的array
    property_names = data[0].dtype.names  # 读取property的名字 ('x', 'y', 'z', 'red', 'green', 'blue')
    for i, name in enumerate(property_names):  # 按property读取数据，这样可以保证读出的数据是同样的数据类型。
        data_np[:, i] = data_pd[name]
    return data_np

class TSDFVolume:
    """Volumetric TSDF Fusion of RGB-D Images.
    """

    def __init__(self, vol_bnds, voxel_size, use_gpu=True, margin=5):
        """Constructor.

        Args:
          vol_bnds (ndarray): An ndarray of shape (3, 2). Specifies the
            xyz bounds (min/max) in meters.
          voxel_size (float): The volume discretization in meters.
        """
        # try:
        FUSION_GPU_MODE = 1
        self.cuda = cuda
        # except Exception as err:
        #     print('Warning: {}'.format(err))
        #     print('Failed to import PyCUDA. Running fusion in CPU mode.')
        #     FUSION_GPU_MODE = 0

        vol_bnds = np.asarray(vol_bnds)
        assert vol_bnds.shape == (3, 2), "[!] `vol_bnds` should be of shape (3, 2)."

        # Define voxel volume parameters
        self._vol_bnds = vol_bnds
        self._voxel_size = float(voxel_size)
        self._trunc_margin = margin * self._voxel_size  # truncation on SDF
        self._color_const = 256 * 256

        # Adjust volume bounds and ensure C-order contiguous C order，C语言风格，即是行序优先，存储方式
        self._vol_dim = np.round((self._vol_bnds[:, 1] - self._vol_bnds[:, 0]) / self._voxel_size).copy(order='C').astype(int)
        self._vol_bnds[:, 1] = self._vol_bnds[:, 0] + self._vol_dim * self._voxel_size
        self._vol_origin = self._vol_bnds[:, 0].copy(order='C').astype(np.float32)

        # Initialize pointers to voxel volume in CPU memory
        self._tsdf_vol_cpu = np.ones(self._vol_dim).astype(np.float32)
        # for computing the cumulative moving average of observations per voxel
        self._weight_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)
        self._color_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)
        # self._index_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)
        # self._color_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)

        self.gpu_mode = use_gpu and FUSION_GPU_MODE

        # Copy voxel volumes to GPU
        if self.gpu_mode:
            self._tsdf_vol_gpu = cuda.mem_alloc(self._tsdf_vol_cpu.nbytes)
            self._weight_vol_gpu = cuda.mem_alloc(self._weight_vol_cpu.nbytes)
            self._color_vol_gpu = cuda.mem_alloc(self._color_vol_cpu.nbytes)
            self.cuda.memcpy_htod(self._tsdf_vol_gpu, self._tsdf_vol_cpu)
            self.cuda.memcpy_htod(self._weight_vol_gpu, self._weight_vol_cpu)
            self.cuda.memcpy_htod(self._color_vol_gpu, self._color_vol_cpu)

            # other_params 
                # gpu_loop_idx,
                # self._voxel_size,
                # im_h,
                # im_w,
                # self._trunc_margin,
                # obs_weight
            # Cuda kernel function (C++)
            self._cuda_src_mod = SourceModule("""
                __global__ void integrate(float * tsdf_vol,
                                        float * weight_vol,
                                        float * color_vol,
                                        float * vol_dim,
                                        float * vol_origin,
                                        float * cam_intr,
                                        float * cam_pose,
                                        float * other_params,
                                        float * color_im,
                                        float * depth_im) {
                // Get voxel index
                int gpu_loop_idx = (int) other_params[0];
                int max_threads_per_block = blockDim.x;
                int block_idx = blockIdx.z*gridDim.y*gridDim.x+blockIdx.y*gridDim.x+blockIdx.x;
                int voxel_idx = gpu_loop_idx*gridDim.x*gridDim.y*gridDim.z*max_threads_per_block+block_idx*max_threads_per_block+threadIdx.x;
                int vol_dim_x = (int) vol_dim[0];
                int vol_dim_y = (int) vol_dim[1];
                int vol_dim_z = (int) vol_dim[2];
                if (voxel_idx > vol_dim_x*vol_dim_y*vol_dim_z)
                    return;
                // Get voxel grid coordinates (note: be careful when casting)
                float voxel_x = floorf(((float)voxel_idx)/((float)(vol_dim_y*vol_dim_z)));
                float voxel_y = floorf(((float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z))/((float)vol_dim_z));
                float voxel_z = (float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z-((int)voxel_y)*vol_dim_z);
                // Voxel grid coordinates to world coordinates
                float voxel_size = other_params[1];
                float pt_x = vol_origin[0]+voxel_x*voxel_size;
                float pt_y = vol_origin[1]+voxel_y*voxel_size;
                float pt_z = vol_origin[2]+voxel_z*voxel_size;
                // World coordinates to camera coordinates
                float tmp_pt_x = pt_x-cam_pose[0*4+3];
                float tmp_pt_y = pt_y-cam_pose[1*4+3];
                float tmp_pt_z = pt_z-cam_pose[2*4+3];
                float cam_pt_x = cam_pose[0*4+0]*tmp_pt_x+cam_pose[1*4+0]*tmp_pt_y+cam_pose[2*4+0]*tmp_pt_z;
                float cam_pt_y = cam_pose[0*4+1]*tmp_pt_x+cam_pose[1*4+1]*tmp_pt_y+cam_pose[2*4+1]*tmp_pt_z;
                float cam_pt_z = cam_pose[0*4+2]*tmp_pt_x+cam_pose[1*4+2]*tmp_pt_y+cam_pose[2*4+2]*tmp_pt_z;
                // Camera coordinates to image pixels
                int pixel_x = (int) roundf(cam_intr[0*3+0]*(cam_pt_x/cam_pt_z)+cam_intr[0*3+2]);
                int pixel_y = (int) roundf(cam_intr[1*3+1]*(cam_pt_y/cam_pt_z)+cam_intr[1*3+2]);
                // Skip if outside view frustum
                int im_h = (int) other_params[2];
                int im_w = (int) other_params[3];
                if (pixel_x < 0 || pixel_x >= im_w || pixel_y < 0 || pixel_y >= im_h || cam_pt_z<0)
                    return;
                // Skip invalid depth
                float depth_value = depth_im[pixel_y*im_w+pixel_x];
                if (depth_value == 0)
                    return;
                // Integrate TSDF
                float trunc_margin = other_params[4];
                float depth_diff = depth_value-cam_pt_z;
                // if (depth_diff < -trunc_margin)
                if (abs(depth_diff) > trunc_margin)
                    return;
                float dist = fmin(1.0f,depth_diff/trunc_margin);
                float w_old = weight_vol[voxel_idx];
                float obs_weight = other_params[5]*(1-abs(dist));
                float w_new = w_old + obs_weight;
                weight_vol[voxel_idx] = w_new;
                tsdf_vol[voxel_idx] = (tsdf_vol[voxel_idx]*w_old+obs_weight*dist)/w_new;
                
                // Integrate color
                // return;
                float old_color = color_vol[voxel_idx];
                float old_b = floorf(old_color/(256*256));
                float old_g = floorf((old_color-old_b*256*256)/256);
                float old_r = old_color-old_b*256*256-old_g*256;
                float new_color = color_im[pixel_y*im_w+pixel_x];
                float new_b = floorf(new_color/(256*256));
                float new_g = floorf((new_color-new_b*256*256)/256);
                float new_r = new_color-new_b*256*256-new_g*256;
                new_b = fmin(roundf((old_b*w_old+obs_weight*new_b)/w_new),255.0f);
                new_g = fmin(roundf((old_g*w_old+obs_weight*new_g)/w_new),255.0f);
                new_r = fmin(roundf((old_r*w_old+obs_weight*new_r)/w_new),255.0f);
                color_vol[voxel_idx] = new_b*256*256+new_g*256+new_r;
                }"""
            )

            self._cuda_integrate = self._cuda_src_mod.get_function("integrate")
            self._cuda_src_mod1 = SourceModule("""
                __global__ void filter(float * tsdf_vol,
                                        float * weight_vol,
                                        float * color_vol,
                                        float * vol_dim,
                                        float * vol_origin,
                                        float * cam_intr,
                                        float * cam_pose,
                                        float * other_params,
                                        float * color_im,
                                        float * depth_im) {
                // Get voxel index
                int gpu_loop_idx = (int) other_params[0];
                int max_threads_per_block = blockDim.x;
                int block_idx = blockIdx.z*gridDim.y*gridDim.x+blockIdx.y*gridDim.x+blockIdx.x;
                int voxel_idx = gpu_loop_idx*gridDim.x*gridDim.y*gridDim.z*max_threads_per_block+block_idx*max_threads_per_block+threadIdx.x;
                int vol_dim_x = (int) vol_dim[0];
                int vol_dim_y = (int) vol_dim[1];
                int vol_dim_z = (int) vol_dim[2];
                if (voxel_idx > vol_dim_x*vol_dim_y*vol_dim_z)
                    return;
                // Get voxel grid coordinates (note: be careful when casting)
                float voxel_x = floorf(((float)voxel_idx)/((float)(vol_dim_y*vol_dim_z)));
                float voxel_y = floorf(((float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z))/((float)vol_dim_z));
                float voxel_z = (float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z-((int)voxel_y)*vol_dim_z);
                // Voxel grid coordinates to world coordinates
                float voxel_size = other_params[1];
                float pt_x = vol_origin[0]+voxel_x*voxel_size;
                float pt_y = vol_origin[1]+voxel_y*voxel_size;
                float pt_z = vol_origin[2]+voxel_z*voxel_size;
                // World coordinates to camera coordinates
                float tmp_pt_x = pt_x-cam_pose[0*4+3];
                float tmp_pt_y = pt_y-cam_pose[1*4+3];
                float tmp_pt_z = pt_z-cam_pose[2*4+3];
                float cam_pt_x = cam_pose[0*4+0]*tmp_pt_x+cam_pose[1*4+0]*tmp_pt_y+cam_pose[2*4+0]*tmp_pt_z;
                float cam_pt_y = cam_pose[0*4+1]*tmp_pt_x+cam_pose[1*4+1]*tmp_pt_y+cam_pose[2*4+1]*tmp_pt_z;
                float cam_pt_z = cam_pose[0*4+2]*tmp_pt_x+cam_pose[1*4+2]*tmp_pt_y+cam_pose[2*4+2]*tmp_pt_z;
                // Camera coordinates to image pixels
                int pixel_x = (int) roundf(cam_intr[0*3+0]*(cam_pt_x/cam_pt_z)+cam_intr[0*3+2]);
                int pixel_y = (int) roundf(cam_intr[1*3+1]*(cam_pt_y/cam_pt_z)+cam_intr[1*3+2]);
                // Skip if outside view frustum
                int im_h = (int) other_params[2];
                int im_w = (int) other_params[3];
                if (pixel_x < 0 || pixel_x >= im_w || pixel_y < 0 || pixel_y >= im_h || cam_pt_z<0)
                    return;
                // Skip invalid depth
                float depth_value = depth_im[pixel_y*im_w+pixel_x];
                if (depth_value == 0)
                    return;
                // Integrate TSDF
                float trunc_margin = other_params[4];
                float depth_diff = depth_value-cam_pt_z;
                // if (depth_diff < -trunc_margin)
                if (abs(depth_diff) > trunc_margin)
                    return;
                // float dist = fmin(1.0f,depth_diff/trunc_margin);
                // float w_old = weight_vol[voxel_idx];
                // float obs_weight = other_params[5];
                // float w_new = w_old + obs_weight;
                // weight_vol[voxel_idx] = w_new;
                // tsdf_vol[voxel_idx] = (tsdf_vol[voxel_idx]*w_old+obs_weight*dist)/w_new;
                
                // Integrate color
                // return;
                float old_color = color_vol[voxel_idx];
                float old_b = floorf(old_color/(256*256));
                float old_g = floorf((old_color-old_b*256*256)/256);
                float old_r = old_color-old_b*256*256-old_g*256;
                float new_color = color_im[pixel_y*im_w+pixel_x];
                float new_b = floorf(new_color/(256*256));
                float new_g = floorf((new_color-new_b*256*256)/256);
                float new_r = new_color-new_b*256*256-new_g*256;
                // float color_gap=sqrt((new_b-old_b)*(new_b-old_b)+(old_g-new_g)*(old_g-new_g)+(old_r-new_r)*(old_r-new_r));
                // if (color_gap>10*sqrt(3))
                //  weight_vol[voxel_idx]=1;
                float color_gap=abs(new_b-old_b)+abs(old_g-new_g)+abs(old_r-new_r);
                if (color_gap>10*5)
                    weight_vol[voxel_idx]=1;
                // new_b = fmin(roundf((old_b*w_old+obs_weight*new_b)/w_new),255.0f);
                // new_g = fmin(roundf((old_g*w_old+obs_weight*new_g)/w_new),255.0f);
                // new_r = fmin(roundf((old_r*w_old+obs_weight*new_r)/w_new),255.0f);
                // color_vol[voxel_idx] = new_b*256*256+new_g*256+new_r;
                }""")
            self._cuda_filter = self._cuda_src_mod1.get_function("filter")
            # Determine block/grid size on GPU
            gpu_dev = cuda.Device(0)
            #* 每个block最大线程数
            self._max_gpu_threads_per_block = gpu_dev.MAX_THREADS_PER_BLOCK
            #* 总共需要的block数
            n_blocks = int(np.ceil(float(np.prod(self._vol_dim)) / float(self._max_gpu_threads_per_block)))
            #* 所有block对应的grid的x，y，z方向的个数
            grid_dim_x = min(gpu_dev.MAX_GRID_DIM_X, int(np.floor(np.cbrt(n_blocks))))
            grid_dim_y = min(gpu_dev.MAX_GRID_DIM_Y, int(np.floor(np.sqrt(n_blocks / grid_dim_x))))
            grid_dim_z = min(gpu_dev.MAX_GRID_DIM_Z, int(np.ceil(float(n_blocks) / float(grid_dim_x * grid_dim_y))))
            self._max_gpu_grid_dim = np.array([grid_dim_x, grid_dim_y, grid_dim_z]).astype(int)
            self._n_gpu_loops = int(np.ceil(float(np.prod(self._vol_dim)) / float(
                np.prod(self._max_gpu_grid_dim) * self._max_gpu_threads_per_block)))

        else:
            # Get voxel grid coordinates
            xv, yv, zv = np.meshgrid(
                range(self._vol_dim[0]),
                range(self._vol_dim[1]),
                range(self._vol_dim[2]),
                indexing='ij'
            )
            self.vox_coords = np.concatenate([
                xv.reshape(1, -1),
                yv.reshape(1, -1),
                zv.reshape(1, -1)
            ], axis=0).astype(int).T

    @staticmethod
    @njit(parallel=True)
    def vox2world(vol_origin, vox_coords, vox_size):
        """Convert voxel grid coordinates to world coordinates.
        """
        vol_origin = vol_origin.astype(np.float32)
        vox_coords = vox_coords.astype(np.float32)
        cam_pts = np.empty_like(vox_coords, dtype=np.float32)
        for i in prange(vox_coords.shape[0]):
            for j in range(3):
                cam_pts[i, j] = vol_origin[j] + (vox_size * vox_coords[i, j])
        return cam_pts

    @staticmethod
    @njit(parallel=True)
    def cam2pix(cam_pts, intr):
        """Convert camera coordinates to pixel coordinates.
        """
        intr = intr.astype(np.float32)
        fx, fy = intr[0, 0], intr[1, 1]
        cx, cy = intr[0, 2], intr[1, 2]
        pix = np.empty((cam_pts.shape[0], 2), dtype=np.int64)
        for i in prange(cam_pts.shape[0]):
            pix[i, 0] = int(np.round((cam_pts[i, 0] * fx / cam_pts[i, 2]) + cx))
            pix[i, 1] = int(np.round((cam_pts[i, 1] * fy / cam_pts[i, 2]) + cy))
        return pix

    @staticmethod
    @njit(parallel=True)
    def integrate_tsdf(tsdf_vol, dist, w_old, obs_weight):
        """Integrate the TSDF volume.
        """
        tsdf_vol_int = np.empty_like(tsdf_vol, dtype=np.float32)
        w_new = np.empty_like(w_old, dtype=np.float32)
        for i in prange(len(tsdf_vol)):
            w_new[i] = w_old[i] + obs_weight
            tsdf_vol_int[i] = (w_old[i] * tsdf_vol[i] + obs_weight * dist[i]) / w_new[i]
        return tsdf_vol_int, w_new

    def integrate(self, color_im, depth_im, cam_intr, cam_pose, obs_weight=1.):
        """Integrate an RGB-D frame into the TSDF volume.

        Args:
          color_im (ndarray): An RGB image of shape (H, W, 3).
          depth_im (ndarray): A depth image of shape (H, W).
          cam_intr (ndarray): The camera intrinsics matrix of shape (3, 3).
          cam_pose (ndarray): The camera pose (i.e. extrinsics) of shape (4, 4).
          obs_weight (float): The weight to assign for the current observation. A higher
            value
        """
        im_h, im_w = depth_im.shape

        if color_im is not None:
            # Fold RGB color image into a single channel image
            color_im = color_im.astype(np.float32)
            color_im = np.floor(color_im[..., 2] * self._color_const + color_im[..., 1] * 256 + color_im[..., 0])
            color_im = color_im.reshape(-1).astype(np.float32)
        else:
            color_im = np.array(0)

        if self.gpu_mode:  # GPU mode: integrate voxel volume (calls CUDA kernel)
            for gpu_loop_idx in range(self._n_gpu_loops):
                self._cuda_integrate(self._tsdf_vol_gpu,
                                     self._weight_vol_gpu,
                                     self._color_vol_gpu,
                                     self.cuda.InOut(self._vol_dim.astype(np.float32)),
                                     self.cuda.InOut(self._vol_origin.astype(np.float32)),
                                     self.cuda.InOut(cam_intr.reshape(-1).astype(np.float32)),
                                     self.cuda.InOut(cam_pose.reshape(-1).astype(np.float32)),
                                     self.cuda.InOut(np.asarray([
                                         gpu_loop_idx,
                                         self._voxel_size,
                                         im_h,
                                         im_w,
                                         self._trunc_margin,
                                         obs_weight
                                     ], np.float32)),
                                     self.cuda.InOut(color_im),
                                     self.cuda.InOut(depth_im.reshape(-1).astype(np.float32)),
                                     block=(self._max_gpu_threads_per_block, 1, 1),
                                     grid=(
                                         int(self._max_gpu_grid_dim[0]),
                                         int(self._max_gpu_grid_dim[1]),
                                         int(self._max_gpu_grid_dim[2]),
                                     )
                                     )
        else:  # CPU mode: integrate voxel volume (vectorized implementation)
            # Convert voxel grid coordinates to pixel coordinates
            cam_pts = self.vox2world(self._vol_origin, self.vox_coords, self._voxel_size)
            cam_pts = rigid_transform(cam_pts, np.linalg.inv(cam_pose))
            pix_z = cam_pts[:, 2]
            pix = self.cam2pix(cam_pts, cam_intr)
            pix_x, pix_y = pix[:, 0], pix[:, 1]

            # Eliminate pixels outside view frustum
            valid_pix = np.logical_and(pix_x >= 0,
                                       np.logical_and(pix_x < im_w,
                                                      np.logical_and(pix_y >= 0,
                                                                     np.logical_and(pix_y < im_h,
                                                                                    pix_z > 0))))
            depth_val = np.zeros(pix_x.shape)
            depth_val[valid_pix] = depth_im[pix_y[valid_pix], pix_x[valid_pix]]

            # Integrate TSDF
            depth_diff = depth_val - pix_z
            valid_pts = np.logical_and(depth_val > 0, depth_diff >= -self._trunc_margin)
            dist = np.minimum(1, depth_diff / self._trunc_margin)
            valid_vox_x = self.vox_coords[valid_pts, 0]
            valid_vox_y = self.vox_coords[valid_pts, 1]
            valid_vox_z = self.vox_coords[valid_pts, 2]
            w_old = self._weight_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
            tsdf_vals = self._tsdf_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
            valid_dist = dist[valid_pts]
            tsdf_vol_new, w_new = self.integrate_tsdf(tsdf_vals, valid_dist, w_old, obs_weight)
            self._weight_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = w_new
            self._tsdf_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = tsdf_vol_new

            # Integrate color
            old_color = self._color_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
            old_b = np.floor(old_color / self._color_const)
            old_g = np.floor((old_color - old_b * self._color_const) / 256)
            old_r = old_color - old_b * self._color_const - old_g * 256
            new_color = color_im[pix_y[valid_pts], pix_x[valid_pts]]
            new_b = np.floor(new_color / self._color_const)
            new_g = np.floor((new_color - new_b * self._color_const) / 256)
            new_r = new_color - new_b * self._color_const - new_g * 256
            new_b = np.minimum(255., np.round((w_old * old_b + obs_weight * new_b) / w_new))
            new_g = np.minimum(255., np.round((w_old * old_g + obs_weight * new_g) / w_new))
            new_r = np.minimum(255., np.round((w_old * old_r + obs_weight * new_r) / w_new))
            self._color_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = new_b * self._color_const + new_g * 256 + new_r

    def filter(self, color_im, depth_im, cam_intr, cam_pose, obs_weight=1.):
        """filter an RGB-D frame into the TSDF volume.

        Args:
          color_im (ndarray): An RGB image of shape (H, W, 3).
          depth_im (ndarray): A depth image of shape (H, W).
          cam_intr (ndarray): The camera intrinsics matrix of shape (3, 3).
          cam_pose (ndarray): The camera pose (i.e. extrinsics) of shape (4, 4).
          obs_weight (float): The weight to assign for the current observation. A higher
            value
        """
        im_h, im_w = depth_im.shape

        if color_im is not None:
            # Fold RGB color image into a single channel image
            color_im = color_im.astype(np.float32)
            color_im = np.floor(color_im[..., 2] * self._color_const + color_im[..., 1] * 256 + color_im[..., 0])
            color_im = color_im.reshape(-1).astype(np.float32)
        else:
            color_im = np.array(0)

        if self.gpu_mode:  # GPU mode: integrate voxel volume (calls CUDA kernel)
            for gpu_loop_idx in range(self._n_gpu_loops):
                self._cuda_filter(self._tsdf_vol_gpu,
                                     self._weight_vol_gpu,
                                     self._color_vol_gpu,
                                     self.cuda.InOut(self._vol_dim.astype(np.float32)),
                                     self.cuda.InOut(self._vol_origin.astype(np.float32)),
                                     self.cuda.InOut(cam_intr.reshape(-1).astype(np.float32)),
                                     self.cuda.InOut(cam_pose.reshape(-1).astype(np.float32)),
                                     self.cuda.InOut(np.asarray([
                                         gpu_loop_idx,
                                         self._voxel_size,
                                         im_h,
                                         im_w,
                                         self._trunc_margin,
                                         obs_weight
                                     ], np.float32)),
                                     self.cuda.InOut(color_im),
                                     self.cuda.InOut(depth_im.reshape(-1).astype(np.float32)),
                                     block=(self._max_gpu_threads_per_block, 1, 1),
                                     grid=(
                                         int(self._max_gpu_grid_dim[0]),
                                         int(self._max_gpu_grid_dim[1]),
                                         int(self._max_gpu_grid_dim[2]),
                                     )
                                     )
        else:  # CPU mode: integrate voxel volume (vectorized implementation)
            # Convert voxel grid coordinates to pixel coordinates
            cam_pts = self.vox2world(self._vol_origin, self.vox_coords, self._voxel_size)
            cam_pts = rigid_transform(cam_pts, np.linalg.inv(cam_pose))
            pix_z = cam_pts[:, 2]
            pix = self.cam2pix(cam_pts, cam_intr)
            pix_x, pix_y = pix[:, 0], pix[:, 1]

            # Eliminate pixels outside view frustum
            valid_pix = np.logical_and(pix_x >= 0,
                                       np.logical_and(pix_x < im_w,
                                                      np.logical_and(pix_y >= 0,
                                                                     np.logical_and(pix_y < im_h,
                                                                                    pix_z > 0))))
            depth_val = np.zeros(pix_x.shape)
            depth_val[valid_pix] = depth_im[pix_y[valid_pix], pix_x[valid_pix]]

            # Integrate TSDF
            depth_diff = depth_val - pix_z
            valid_pts = np.logical_and(depth_val > 0, depth_diff >= -self._trunc_margin)
            dist = np.minimum(1, depth_diff / self._trunc_margin)
            valid_vox_x = self.vox_coords[valid_pts, 0]
            valid_vox_y = self.vox_coords[valid_pts, 1]
            valid_vox_z = self.vox_coords[valid_pts, 2]
            w_old = self._weight_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
            tsdf_vals = self._tsdf_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
            valid_dist = dist[valid_pts]
            tsdf_vol_new, w_new = self.integrate_tsdf(tsdf_vals, valid_dist, w_old, obs_weight)
            self._weight_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = w_new
            self._tsdf_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = tsdf_vol_new

            # Integrate color
            old_color = self._color_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
            old_b = np.floor(old_color / self._color_const)
            old_g = np.floor((old_color - old_b * self._color_const) / 256)
            old_r = old_color - old_b * self._color_const - old_g * 256
            new_color = color_im[pix_y[valid_pts], pix_x[valid_pts]]
            new_b = np.floor(new_color / self._color_const)
            new_g = np.floor((new_color - new_b * self._color_const) / 256)
            new_r = new_color - new_b * self._color_const - new_g * 256
            new_b = np.minimum(255., np.round((w_old * old_b + obs_weight * new_b) / w_new))
            new_g = np.minimum(255., np.round((w_old * old_g + obs_weight * new_g) / w_new))
            new_r = np.minimum(255., np.round((w_old * old_r + obs_weight * new_r) / w_new))
            self._color_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = new_b * self._color_const + new_g * 256 + new_r

    def get_volume(self):
        if self.gpu_mode:
            self.cuda.memcpy_dtoh(self._tsdf_vol_cpu, self._tsdf_vol_gpu)
            self.cuda.memcpy_dtoh(self._color_vol_cpu, self._color_vol_gpu)
            self.cuda.memcpy_dtoh(self._weight_vol_cpu, self._weight_vol_gpu)
        self._tsdf_vol_cpu[self._weight_vol_cpu<3]=1
        # np.nonzero(self._tsdf_vol_cpu<1)[0].shape
        # import pdb;pdb.set_trace()
        return self._tsdf_vol_cpu, self._color_vol_cpu, self._weight_vol_cpu

    def get_point_cloud(self):
        """Extract a point cloud from the voxel volume.
        """
        tsdf_vol, color_vol, weight_vol = self.get_volume()
        # Marching cubes
        # verts = measure.marching_cubes_lewiner(tsdf_vol, level=0)[0]
        verts = measure.marching_cubes(tsdf_vol, level=0)[0]
        verts_ind = np.round(verts).astype(int)
        verts = verts * self._voxel_size + self._vol_origin
        # Get vertex colors
        rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
        colors_b = np.floor(rgb_vals / self._color_const)
        colors_g = np.floor((rgb_vals - colors_b * self._color_const) / 256)
        colors_r = rgb_vals - colors_b * self._color_const - colors_g * 256
        colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
        colors = colors.astype(np.uint8)
        pc = np.hstack([verts, colors])
        return pc

    def get_mesh(self):
        """Compute a mesh from the voxel volume using marching cubes.
        """
        tsdf_vol, color_vol, weight_vol = self.get_volume()
        # verts, faces, norms, vals = measure.marching_cubes_lewiner(tsdf_vol, level=0)
        # import pdb;pdb.set_trace()
        verts, faces, norms, vals = measure.marching_cubes(tsdf_vol, level=0)
        verts_ind = np.round(verts).astype(int)
        verts = verts * self._voxel_size + self._vol_origin  # voxel grid coordinates to world coordinates
        # Get vertex colors
        rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
        colors_b = np.floor(rgb_vals / self._color_const)
        colors_g = np.floor((rgb_vals - colors_b * self._color_const) / 256)
        colors_r = rgb_vals - colors_b * self._color_const - colors_g * 256
        colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
        colors = colors.astype(np.uint8)
        return verts, faces, norms, colors

def rigid_transform(xyz, transform):
    """Applies a rigid transform to an (N, 3) pointcloud.
    """
    xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])
    xyz_t_h = np.dot(transform, xyz_h.T).T
    return xyz_t_h[:, :3]

def get_view_frustum(depth_im, cam_intr, cam_pose):
    """Get corners of 3D camera view frustum of depth image
    """
    im_h = depth_im.shape[0]
    im_w = depth_im.shape[1]
    max_depth = np.max(depth_im)
    #* 由最大深度确定的视锥的五个顶点（相机坐标系下）
    view_frust_pts = np.array([
        (np.array([0, 0, 0, im_w, im_w]) - cam_intr[0, 2]) * np.array([0, max_depth, max_depth, max_depth, max_depth]) /
        cam_intr[0, 0],
        (np.array([0, 0, im_h, 0, im_h]) - cam_intr[1, 2]) * np.array([0, max_depth, max_depth, max_depth, max_depth]) /
        cam_intr[1, 1],
        np.array([0, max_depth, max_depth, max_depth, max_depth])
    ])
    #* 转到全局坐标系
    view_frust_pts = rigid_transform(view_frust_pts.T, cam_pose).T
    return view_frust_pts

def meshwrite(filename, verts, faces, norms, colors):
    """Save a 3D mesh to a polygon .ply file.
    """
    # Write header
    ply_file = open(filename, 'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (verts.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property float nx\n")
    ply_file.write("property float ny\n")
    ply_file.write("property float nz\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("element face %d\n" % (faces.shape[0]))
    ply_file.write("property list uchar int vertex_index\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(verts.shape[0]):
        ply_file.write("%f %f %f %f %f %f %d %d %d\n" % (
            verts[i, 0], verts[i, 1], verts[i, 2],
            norms[i, 0], norms[i, 1], norms[i, 2],
            colors[i, 0], colors[i, 1], colors[i, 2],
        ))

    # Write face list
    for i in range(faces.shape[0]):
        ply_file.write("3 %d %d %d\n" % (faces[i, 0], faces[i, 1], faces[i, 2]))

    ply_file.close()

def pcwrite(filename, xyzrgb):
    """Save a point cloud to a polygon .ply file.
    """
    xyz = xyzrgb[:, :3]
    rgb = xyzrgb[:, 3:].astype(np.uint8)

    # Write header
    ply_file = open(filename, 'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (xyz.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(xyz.shape[0]):
        ply_file.write("%f %f %f %d %d %d\n" % (
            xyz[i, 0], xyz[i, 1], xyz[i, 2],
            rgb[i, 0], rgb[i, 1], rgb[i, 2],
        ))

def integrate(
        depth_im,
        cam_intr,
        cam_pose,
        obs_weight,
        world_c,
        vox_coords,
        weight_vol,
        tsdf_vol,
        sdf_trunc,
        im_h,
        im_w,
):
    # Convert world coordinates to camera coordinates
    world2cam = torch.inverse(cam_pose)
    cam_c = torch.matmul(world2cam, world_c.transpose(1, 0)).transpose(1, 0).float()

    # Convert camera coordinates to pixel coordinates
    fx, fy = cam_intr[0, 0], cam_intr[1, 1]
    cx, cy = cam_intr[0, 2], cam_intr[1, 2]
    pix_z = cam_c[:, 2]
    pix_x = torch.round((cam_c[:, 0] * fx / cam_c[:, 2]) + cx).long()
    pix_y = torch.round((cam_c[:, 1] * fy / cam_c[:, 2]) + cy).long()

    # Eliminate pixels outside view frustum
    valid_pix = (pix_x >= 0) & (pix_x < im_w) & (pix_y >= 0) & (pix_y < im_h) & (pix_z > 0)
    valid_vox_x = vox_coords[valid_pix, 0]
    valid_vox_y = vox_coords[valid_pix, 1]
    valid_vox_z = vox_coords[valid_pix, 2]
    depth_val = depth_im[pix_y[valid_pix], pix_x[valid_pix]]

    # Integrate tsdf
    depth_diff = depth_val - pix_z[valid_pix]
    dist = torch.clamp(depth_diff / sdf_trunc, max=1)
    valid_pts = (depth_val > 0) & (depth_diff >= -sdf_trunc)
    valid_vox_x = valid_vox_x[valid_pts]
    valid_vox_y = valid_vox_y[valid_pts]
    valid_vox_z = valid_vox_z[valid_pts]
    valid_dist = dist[valid_pts]
    w_old = weight_vol[valid_vox_x, valid_vox_y, valid_vox_z]
    tsdf_vals = tsdf_vol[valid_vox_x, valid_vox_y, valid_vox_z]
    w_new = w_old + obs_weight
    tsdf_vol[valid_vox_x, valid_vox_y, valid_vox_z] = (w_old * tsdf_vals + obs_weight * valid_dist) / w_new
    weight_vol[valid_vox_x, valid_vox_y, valid_vox_z] = w_new

    return weight_vol, tsdf_vol

def read_img( filepath):
    img = np.array(Image.open(filepath))
    return img

def read_depth( filepath, scaler = 1000.0):
    # Read depth image and camera pose
    depth_im = cv2.imread(filepath, -1).astype(np.float32)
    depth_im /= scaler  # depth is saved in 16-bit PNG in millimeters
    depth_im[depth_im > 3.0] = 0
    return depth_im

def fuse_depth2ply(datapath, num_depth=-1 ,scalingFactor=1000.0):
    intrinsic_datapath = os.path.join(datapath,'intrinsic','intrinsic.txt')
    image_filenames = os.listdir(os.path.join(datapath,'images'))
    image_filenames.sort(key=lambda f : int(f.split('.')[0]))
    image_datapath = [os.path.join(datapath,'images',f) for f in image_filenames]
    extrinsic_datapath = [os.path.join(datapath,'extrinsic',f.replace('.png','.txt')) for f in image_filenames]
    depth_datapath = [os.path.join(datapath,'depth',f) for f in image_filenames]
    K = np.loadtxt(intrinsic_datapath)
    K_inv = np.linalg.inv(K)
    all_pts = []
    num_d = 0
    for i in tqdm(range(len(extrinsic_datapath)), desc="Fusing depth..."):
        if num_depth>0 and i>num_depth:
            break
        extrinsic = np.loadtxt(extrinsic_datapath[i])
        pose = np.linalg.inv(extrinsic)
        # pose = extrinsic
        rgb = cv2.imread(image_datapath[i])[:,:,[2,1,0]]
        # import pdb;pdb.set_trace()
        depth = cv2.imread(depth_datapath[i],-1)/scalingFactor
        valid_mask = depth>0
        height, width,_ = rgb.shape
        xx,yy = np.meshgrid(np.arange(width),np.arange(height))
        uv1 = np.stack([xx,yy,np.ones_like(xx)],axis=2)
        uvd = uv1*depth[...,None] #* (480, 640, 3)
        uvd_valid = uvd[valid_mask,:] #* (307200, 3)
        rgb_valid = rgb[valid_mask]
        uvd_valid = uvd_valid.T #* (3, 307200)
        xyz_c = K_inv @ uvd_valid
        xyz_w = pose[:3,:3] @ xyz_c + pose[:3,3][...,None] #* (3, 307200)
        xyzcolor = np.concatenate([xyz_w.T, rgb_valid],axis=1) #* (307200, 6)
        #* xyzcolor.shape (696382, 6)
        #* sparse_volume.size() 4032981 因为点与点之间的距离较大，而网格尺寸较小，因此涉及的网格的数量比点还多
        all_pts.append(xyzcolor)
        num_d+=1
    all_pts_np = np.concatenate(all_pts,axis=0)
    save_path = os.path.join(datapath,'ply')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pcwrite(os.path.join(save_path, f'point_cloud_{num_d}_d.ply'), all_pts_np)

def o3d_tsdf_fusion(datapath, num_depth=-1, voxel_size=0.02, scalingFactor=1000.0):
    intrinsic_datapath = os.path.join(datapath,'intrinsic','intrinsic.txt')
    image_filenames = os.listdir(os.path.join(datapath,'images'))
    image_filenames.sort(key=lambda f : int(f.split('.')[0]))
    image_datapath = [os.path.join(datapath,'images',f) for f in image_filenames]
    extrinsic_datapath = [os.path.join(datapath,'extrinsic',f.replace('.png','.txt')) for f in image_filenames]
    depth_datapath = [os.path.join(datapath,'depth',f) for f in image_filenames]
    K = np.loadtxt(intrinsic_datapath)
    o3d_tsdf = TSDFFusion(voxel_size=voxel_size)
    num_d = 0
    for i in tqdm(range(len(extrinsic_datapath)), desc="O3d fusing depth..."):
        if num_depth>0 and i>num_depth:
            break
        extrinsic = np.loadtxt(extrinsic_datapath[i])
        pose = np.linalg.inv(extrinsic)
        rgb = cv2.imread(image_datapath[i])[:,:,[2,1,0]]
        depth = cv2.imread(depth_datapath[i],-1)/scalingFactor
        o3d_tsdf.integrate(depth, rgb, pose, K)
        num_d+=1
    save_path = os.path.join(datapath,'ply')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    o3d_tsdf.marching_cube(path=os.path.join(save_path, f'o3d_{num_d}_d.ply'))

def tsdf_fusion(datapath, save_mesh=True, scalingFactor=1000.0):
    vol_bnds = np.zeros((3, 2))
    intrinsic_datapath = os.path.join(datapath,'intrinsic','intrinsic.txt')
    image_filenames = os.listdir(os.path.join(datapath,'images'))
    image_filenames.sort(key=lambda f : int(f.split('.')[0]))
    image_datapath = [os.path.join(datapath,'images',f) for f in image_filenames]
    extrinsic_datapath = [os.path.join(datapath,'extrinsic',f.replace('.png','.txt')) for f in image_filenames]
    depth_datapath = [os.path.join(datapath,'depth_comp',f) for f in image_filenames]
    K = np.loadtxt(intrinsic_datapath)
    o3d_tsdf_fusion = TSDFFusion(voxel_size=0.02)
    sparse_volume = SparseVolume(n_feats=3, voxel_size=0.05, dimensions=np.asarray([200.0,200.0,200.0]))
    # import pdb;pdb.set_trace()
    K_inv = np.linalg.inv(K)
    depthes = []
    rgbs = []
    poses = []
    all_pts = []
    for i in tqdm(range(len(extrinsic_datapath)), desc="Fusing depth..."):
        extrinsic = np.loadtxt(extrinsic_datapath[i])
        pose = np.linalg.inv(extrinsic)
        # pose = extrinsic
        rgb = cv2.imread(image_datapath[i])[:,:,[2,1,0]]
        # import pdb;pdb.set_trace()
        depth = cv2.imread(depth_datapath[i],-1)/scalingFactor
        valid_mask = depth>0
        height, width,_ = rgb.shape
        xx,yy = np.meshgrid(np.arange(width),np.arange(height))
        uv1 = np.stack([xx,yy,np.ones_like(xx)],axis=2)
        uvd = uv1*depth[...,None] #* (480, 640, 3)
        uvd_valid = uvd[valid_mask,:] #* (307200, 3)
        rgb_valid = rgb[valid_mask]
        uvd_valid = uvd_valid.T #* (3, 307200)
        xyz_c = K_inv @ uvd_valid
        xyz_w = pose[:3,:3] @ xyz_c + pose[:3,3][...,None] #* (3, 307200)
        xyzcolor = np.concatenate([xyz_w.T, rgb_valid],axis=1) #* (307200, 6)
        #* xyzcolor.shape (696382, 6)
        #* sparse_volume.size() 4032981 因为点与点之间的距离较大，而网格尺寸较小，因此涉及的网格的数量比点还多
        sparse_volume.integrate_pts(xyzcolor)
        import pdb;pdb.set_trace()
        all_pts.append(xyzcolor)
        o3d_tsdf_fusion.integrate(depth, rgb, pose, K)
        depthes.append(depth)
        rgbs.append(rgb)
        poses.append(pose)
        # Compute camera view frustum and extend convex hull
        view_frust_pts = get_view_frustum(depth, K, pose) #* 获得当前图片的全局坐标系视锥的五个顶点（3，5）
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
    all_pts_np = np.concatenate(all_pts,axis=0)
    # pcwrite(os.path.join(datapath,'ply','point_cloud.ply'), all_pts_np)
    o3d_tsdf_fusion.marching_cube(path=os.path.join(datapath, 'ply', 'o3d.ply'))
    import pdb;pdb.set_trace()
    # Initialize voxel volume
    print("Initializing voxel volume...")
    tsdf_vol = TSDFVolume(vol_bnds, voxel_size=0.01 * 2 ** 3, margin=3)
    # Loop through RGB-D images and fuse them together
    for i in tqdm(range(len(poses)), desc="integrate depth..."):
        tsdf_vol.integrate(rgbs[i], depthes[i], K, poses[i], obs_weight=1.)
    if save_mesh:
        print("Saving mesh to mesh.ply...")
        verts, faces, norms, colors = tsdf_vol.get_mesh()
        meshwrite(os.path.join(datapath,'ply',  'mesh_layer.ply'), verts, faces, norms, colors)
        # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
        print("Saving point cloud to pc.ply...",os.path.join(datapath,'ply',  'mesh_layer.ply'))
        point_cloud = tsdf_vol.get_point_cloud()
        pcwrite(os.path.join(datapath, 'ply', 'pc_layer.ply'), point_cloud)

def test_from_np2ply2mesh(datapath):
    #* from numpy to plt to mesh
    x = np.linspace(-3, 3, 401)
    mesh_x, mesh_y = np.meshgrid(x, x)
    z = np.sinc((np.power(mesh_x, 2) + np.power(mesh_y, 2)))
    z_norm = (z - z.min()) / (z.max() - z.min())
    xyz = np.zeros((np.size(mesh_x), 3))
    xyz[:, 0] = np.reshape(mesh_x, -1)
    xyz[:, 1] = np.reshape(mesh_y, -1)
    xyz[:, 2] = np.reshape(z_norm, -1)
    colormap = plt.get_cmap('hsv')
    colors = colormap(xyz[:, 2])[:,:3]

    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # pcd.normals = o3d.utility.Vector3dVector(nxnynz)
    o3d.io.write_point_cloud(datapath + "/sync.ply", pcd)

    # Load saved point cloud and visualize it
    o3d.visualization.draw_geometries([pcd])
    radius = 0.01   # 搜索半径
    max_nn = 30     # 邻域内用于估算法线的最大点数
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))     # 执行法线估计
    radii = [0.01, 0.01, 0.01, 0.04]
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
    o3d.visualization.draw_geometries([pcd, rec_mesh])
    o3d.io.write_triangle_mesh(datapath + "/sync_mesh.ply", rec_mesh)

if __name__=="__main__":
    datapath = '/home/zhujun/catkin_ws/src/r3live-master/r3live_output/data_for_mesh'
    # tsdf_fusion(datapath)
    fuse_depth2ply(datapath,10)