import numpy as np
import open3d as o3d
import open3d.core as o3c
from skimage.measure import marching_cubes
import torch
import torch.nn.functional as F
import torch.utils.dlpack
import trimesh

import tsdf_fusion.utils.voxel_utils as voxel_utils
import tsdf_fusion.utils.o3d_helper as o3d_helper
from torch_scatter import scatter_mean
torch.ops.load_library("/media/zhujun/0DFD06D20DFD06D2/MVS/FrameMVS/ACMMP_net/build/libsimpletools.so")
simpletools = torch.ops.simple_tools
import time
from utils import get_neighbors


class SparseVolume:
    def __init__(self, n_feats=3, voxel_size=0.01, dimensions=np.asarray([10.0,10.0,10.0]), min_pts_in_grid=8, capacity=100000, device="cpu:0") -> None:
        """
        Based on hash map implementation in Open3D.  device="cuda:0"
        """
        min_coords, max_coords, n_xyz = voxel_utils.get_world_range(dimensions, voxel_size)
        print("Creating hashmap!")
        print('min_coords: ',min_coords)
        print('max_coords: ',max_coords)
        print('n_xyz: ',n_xyz)
        self.device = device
        self.dimensions = dimensions #* 5.94939012825489 2.6250001192092896 2.9523400962352753
        self.voxel_size = voxel_size #* 0.01
        self.o3c_device = o3c.Device(device)
        self.min_coords = torch.from_numpy(min_coords).float().to(device)
        self.max_coords = torch.from_numpy(max_coords).float().to(device)
        self.n_xyz = torch.from_numpy(np.asarray(n_xyz)).long().to(device)
        self.n_feats = n_feats #* 8
        self.min_pts_in_grid = min_pts_in_grid #* 8
        self.reset(capacity) #* 100000

        self.avg_n_pts = 0 #* 所有帧的平均点数
        self.n_pts_list = [] #* 每帧的有效格子的平均点数
        self.n_frames = 0 #* 帧数
        self.min_pts = 1000 #* 最小平均点数
        self.max_pts = 0 #* 最大平均点数
    
    def size(self):
        return self.indexer.size()
    
    def reset(self, capacity):
        self.indexer = o3c.HashMap(
            capacity,
            key_dtype=o3c.int64,
            key_element_shape=(3,),
            #* key_element_shape == (), keys.shape == (N,); 
            #* key_element_shape == (3,), keys.shape == (N, 3).; 
            #* key_element_shape == (8, 8, 8), keys.shape == (N, 8, 8, 8).
            value_dtypes=(o3c.Dtype.Float32, o3c.Dtype.Float32, o3c.Dtype.Float32),
            value_element_shapes=((self.n_feats,), (1,), (1,)),
            #* val_elment_shape == (), vals.shape == (N,); 
            #* val_elment_shape == (3,), vals.shape == (N, 3)
            #* val_elment_shape == (8, 8, 8), vals.shape == (N, 8, 8, 8).
            device=self.o3c_device)            
        # to be initialized in self.to_tensor
        self.tensor_indexer = None
        self.features = None
        self.weights = None
        self.num_hits = None
        self.active_coordinates = None

    def to_tensor(self):
        """ store all active values to pytorch tensor
        *将激活值全部转为pytorch tensor
        """
        active_buf_indices = self.indexer.active_buf_indices().to(o3c.int64) #* 激活的值应该就是有数据的网格
        capacity = len(active_buf_indices)
        #* 用于查找key在tensor中的index， hashmap可以快速查找，便于快速更新key对应的值
        self.tensor_indexer = o3c.HashMap(
            capacity,
            key_dtype=o3c.int64,
            key_element_shape=(3,),
            value_dtype=o3c.int64,
            value_element_shape=(1,),
            device=o3c.Device(self.device)
        )
        active_keys = self.indexer.key_tensor()[active_buf_indices].to(o3c.int64)
        #* self.indexer.key_tensor().shape SizeVector[100000, 3]  包含全部数据
        #* self.indexer.active_buf_indices().shape SizeVector[45232] 包含有值的数据
        features = self.indexer.value_tensor(0)[active_buf_indices]
        weights = self.indexer.value_tensor(1)[active_buf_indices]
        num_hits = self.indexer.value_tensor(2)[active_buf_indices]
        indexer_value = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(torch.arange(capacity, device=self.device)))
        buf_indices, masks = self.tensor_indexer.insert(active_keys, indexer_value)
        masks = masks.cpu().numpy() if "cuda" in self.device else masks.numpy()
        assert masks.all()
        self.active_coordinates = torch.utils.dlpack.from_dlpack(active_keys.to_dlpack()) #* 坐标
        self.features = torch.utils.dlpack.from_dlpack(features.to_dlpack()) #* 颜色
        self.weights = torch.utils.dlpack.from_dlpack(weights.to_dlpack()) #* 权重
        self.num_hits = torch.utils.dlpack.from_dlpack(num_hits.to_dlpack()) #* 访问次数
        # import pdb;pdb.set_trace()
        return self.active_coordinates, self.features, self.weights, self.num_hits

    def to_pc(self,):
        cur_pts_coord = self.active_coordinates.float()
        cur_pts = cur_pts_coord * self.voxel_size + self.min_coords.float()[None]
        pts_all = torch.cat([cur_pts, self.features],dim=1)
        return pts_all.numpy()

    def proj2depth(self, ext, K, width, height):
        #* 将激活点坐标投影到图像上
        # import pdb;pdb.set_trace()
        depth = torch.ones((height,width)).float().cuda() * 1000.0
        cur_pts_coord = self.active_coordinates.float().cuda() #* torch.Size([1575607, 3])
        cur_pts = cur_pts_coord * self.voxel_size + self.min_coords.float().cuda()[None]
        pts_c_all = ext[:3,:3] @ cur_pts.T + ext[:3,3:4] #* torch.Size([3, 1575607])
        valid_mask = pts_c_all[2,:]>0.2
        pts_c = pts_c_all[:,valid_mask] #* torch.Size([3, 1575607])
        uvz = K @ pts_c
        z = uvz[2,:]
        uv = uvz[:2,:]/z
        uv_valid_mask = (uv[0,:]>0)*(uv[0,:]<width-1)*(uv[1,:]>0)*(uv[1,:]<height-1)
        uv_valid = uv[:,uv_valid_mask].T #* torch.Size([665540, 2])
        z_valid = z[uv_valid_mask]
        # start = time.time()
        # for uv_i in range(len(uv_valid)):
        #     u,v = int(uv_valid[uv_i,0]), int(uv_valid[uv_i,1])
        #     if depth[v,u]>z_valid[uv_i]:
        #         depth[v,u]=z_valid[uv_i]
        # end1 = time.time()
        # torch.ops.load_library("/home/zhujun/MVS/FrameMVS/ACMMP_net/build/libsimpletools.so")
        # simpletools = torch.ops.simple_tools
        depth1 = simpletools.proj2depth(uv_valid, z_valid, depth)
        # end2 = time.time()
        # print(end1-start,", ",end2-end1) #* 28.807692289352417 ,  0.023688316345214844
        # import pdb;pdb.set_trace()
        depth1[depth1>=500] = 0.0
        # depth_np = depth1.numpy()
        return depth1

    def integrate_pts(self,pts):
        if isinstance(pts,np.ndarray):
            pts = torch.from_numpy(pts).float().to(self.device)
        else:
            pts =pts.float().to(self.device)
        #* pts [N, 3+]
        #* 找到这些点对应的网格
        xyz = pts[:,:3] #* [N, 3]
        point_feats = pts[:,3:].T[None,...] #* [1, c, N]
        ch, N = point_feats.shape[1:]
        xyz_zeroed = xyz - self.min_coords
        xyz_normalized = xyz_zeroed / self.voxel_size #* 归一化坐标 [N, 3]
        # import pdb;pdb.set_trace()
        #* 计算周围8个点的整数坐标
        # grid_id = get_neighbors(xyz_normalized[None,None]).squeeze() 
        # relative_xyz, grid_id = self.get_relative_xyz(xyz) #* 返回当前点相对于周围8个网格点的实际坐标 [8, N, 3]
        grid_id = xyz_normalized.long().reshape(-1, 3)
        point_feats = point_feats[...,None].reshape(1,ch,-1)
        flat_ids = voxel_utils.flatten(grid_id, self.n_xyz).long()  #* [1, N] 三维转化为单个index
        unique_flat_ids, pinds, pcounts = torch.unique(flat_ids, return_inverse=True, return_counts=True)
        #* pinds 为原始数据在unique_flat_ids中的索引
        #* pcounts 不同元素的个数
        # import pdb;pdb.set_trace()
        if len(unique_flat_ids) == 0:
            return
        unique_grid_ids = voxel_utils.unflatten(unique_flat_ids, self.n_xyz).long() #* 一维还原为三维 [N1, 3]
        point_feats_mean = scatter_mean(point_feats, pinds.unsqueeze(0).unsqueeze(0)) #* torch.Size([1, c, N1])
        
        #* 对同一个格子里的形状编码求均值
        #* point_feats       torch.Size([1, c, N]) 
        #* point_feats_mean  torch.Size([1, c, N1])
        #* scatter方法通过src和index两个张量来获得一个新的张量。
        #* 根据index，将index相同值对应的src元素进行对应定义的计算，dim为在第几维进行相应的运算。e.g.scatter_sum即进行sum运算，scatter_mean即进行mean运算。
        fine_feats = point_feats_mean.squeeze().T[:,:3]
        # pts_test = torch.cat([unique_grid_ids, fine_feats],dim=1).numpy()
        # from utils import pcwrite
        # pcwrite(f"output/ply/xyzrgb_{2}.ply", pts_test)
        # pcwrite(f"output/ply/xyzrgb_{2}.ply", pts_test)

        # import pdb;pdb.set_trace()
        fine_weights = torch.ones((fine_feats.shape[0],1)).type_as(fine_feats)
        fine_num_hits = torch.ones((fine_feats.shape[0],1)).type_as(fine_feats)
        if self.indexer.size() > 0:
            model_feats, model_weights, model_num_hits = self._query_tensor(unique_grid_ids)
            new_fine_weights = model_weights + fine_weights
            new_fine_feats = (model_feats * model_weights + fine_feats * fine_weights) / new_fine_weights
            new_fine_num_hits = model_num_hits + fine_num_hits
            # pts_test = torch.cat([unique_grid_ids, new_fine_feats],dim=1).numpy()
            # from utils import pcwrite
            # pcwrite(f"output/ply/xyzrgb_{2}.ply", pts_test)
            # import pdb;pdb.set_trace()
            self.insert(unique_grid_ids, new_fine_feats, new_fine_weights, new_fine_num_hits)
            self.to_tensor()
        else:
            pts_test = torch.cat([unique_grid_ids, fine_feats],dim=1).numpy()
            from utils import pcwrite
            pcwrite(f"output/ply/xyzrgb_{2}.ply", pts_test)
            import pdb;pdb.set_trace()
            self.insert(unique_grid_ids, fine_feats, fine_weights, fine_num_hits)
            self.to_tensor()
        # pts_test = torch.cat([unique_grid_ids, fine_feats],dim=1).numpy()
        # from utils import pcwrite
        # pcwrite(f"output/ply/xyzrgb_{2}.ply", pts_test)
        # pcwrite(f"output/ply/xyzrgb_{2}.ply", pts_test)
        # cur_pts_coord = self.active_coordinates.float()
        # cur_pts = cur_pts_coord * self.voxel_size + self.min_coords.float()[None]
        # pts_all = torch.cat([cur_pts, self.features],dim=1)

    def integrate_pts_complex(self,pts):
        if isinstance(pts,np.ndarray):
            pts = torch.from_numpy(pts).float().to(self.device)
        else:
            pts =pts.float().to(self.device)
        #* pts [N, 3+]
        #* 找到这些点对应的网格
        xyz = pts[:,:3] #* [N, 3]
        point_feats = pts[:,3:].T[None,...] #* [1, c, N]
        ch, N = point_feats.shape[1:]
        relative_xyz, grid_id = self.get_relative_xyz(xyz) #* 返回当前点相对于周围8个网格点的实际坐标 [8, N, 3]
        grid_id = grid_id.reshape(-1, 3)
        point_feats = point_feats[...,None].repeat(1,1,1,8).reshape(1,ch,-1)
        flat_ids = voxel_utils.flatten(grid_id, self.n_xyz).long()  #* [1, N] 三维转化为单个index
        unique_flat_ids, pinds, pcounts = torch.unique(flat_ids, return_inverse=True, return_counts=True)
        #* pinds 为原始数据在unique_flat_ids中的索引
        #* pcounts 不同元素的个数
        # import pdb;pdb.set_trace()
        if len(unique_flat_ids) == 0:
            return
        unique_grid_ids = voxel_utils.unflatten(unique_flat_ids, self.n_xyz).long() #* 一维还原为三维 [N1, 3]
        point_feats_mean = scatter_mean(point_feats, pinds.unsqueeze(0).unsqueeze(0)) #* torch.Size([1, c, N1])
        # import pdb;pdb.set_trace()
        #* 对同一个格子里的形状编码求均值
        #* point_feats       torch.Size([1, c, N]) 
        #* point_feats_mean  torch.Size([1, c, N1])
        #* scatter方法通过src和index两个张量来获得一个新的张量。
        #* 根据index，将index相同值对应的src元素进行对应定义的计算，dim为在第几维进行相应的运算。e.g.scatter_sum即进行sum运算，scatter_mean即进行mean运算。
        fine_feats = point_feats_mean.squeeze().T[:,:3]
        fine_weights = torch.ones((fine_feats.shape[0],1)).type_as(fine_feats)
        fine_num_hits = torch.ones((fine_feats.shape[0],1)).type_as(fine_feats)
        if self.indexer.size() > 0:
            model_feats, model_weights, model_num_hits = self._query_tensor(unique_grid_ids)
            new_fine_weights = model_weights + fine_weights
            new_fine_feats = (model_feats * model_weights + fine_feats * fine_weights) / new_fine_weights
            new_fine_num_hits = model_num_hits + fine_num_hits
            self.insert(unique_grid_ids, new_fine_feats, new_fine_weights, new_fine_num_hits)
            self.to_tensor()
        else:
            self.insert(unique_grid_ids, fine_feats, fine_weights, fine_num_hits)
            self.to_tensor()

    def get_relative_xyz(self,xyz):
        #* xyz [N, 3]
        xyz_zeroed = xyz - self.min_coords
        xyz_normalized = xyz_zeroed / self.voxel_size #* 归一化坐标 [N, 3]
        # import pdb;pdb.set_trace()
        #* 计算周围8个点的整数坐标
        grid_id = get_neighbors(xyz_normalized[None,None]).squeeze()  #* [8, N, 3] 三个坐标，向上取整或者向下取整，8种组合
        relative_xyz_normalized = xyz_normalized.unsqueeze(0) - grid_id #* [8, N, 3]
        relative_xyz = relative_xyz_normalized * self.voxel_size
        return relative_xyz, grid_id.long() #* 当前点相对于网格8个顶点的坐标和8个顶点的网格坐标
    
    #* 覆盖写入
    def insert(self, keys, new_feats, new_weights, new_num_hits):
        if len(keys) == 0:
            return None
        o3c_keys = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(keys)).to(o3c.int64)
        feats_o3c = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(new_feats))
        weights_o3c = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(new_weights))
        num_hits_o3c = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(new_num_hits))
        buf_indices, masks_insert = self.indexer.insert(o3c_keys, (feats_o3c, weights_o3c, num_hits_o3c))
        #* masks indicates whether a (key, value) pair is successfully inserted. 
        #* A mask of value True means the insertion is successful and False if the insertion is skipped.
        #* Unsuccessful insertion only happens when there are duplicated keys.
        #* That is, for a set of duplicated keys, one and only one will get a True mask.
        #* Since the insertion runs in parallel, there is no guarantee which one of the duplicated keys will be inserted. 
        #* That is, for a set of duplicated keys, we don’t know which key gets the True mask.
        if not masks_insert.cpu().numpy().all() and 0:
            existed_masks = masks_insert == False #* 万一有多个false呢？
            existed_keys = o3c_keys[existed_masks]
            buf_indices, masks_find = self.indexer.find(existed_keys)
            assert masks_find.cpu().numpy().all()
            self.indexer.value_tensor(0)[buf_indices.to(o3c.int64)] = feats_o3c[existed_masks]
            self.indexer.value_tensor(1)[buf_indices.to(o3c.int64)] = weights_o3c[existed_masks]
            self.indexer.value_tensor(2)[buf_indices.to(o3c.int64)] = num_hits_o3c[existed_masks]

    def count_optim(self, keys):
        """[summary]
        Args:
            keys ([torch.Tensor]): shape: [1, 8, B, N, 3]  torch.Size([1, 8, 1000, 35, 3])
        Returns:
            [type]: [description]
        """
        shapes = [s for s in keys.shape]
        assert shapes[-1] == 3
        o3c_keys = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(keys.reshape(-1, 3).long()))
        buf_indices, masks = self.tensor_indexer.find(o3c_keys)
        buf_indices = buf_indices[masks]
        indices = self.tensor_indexer.value_tensor()[buf_indices]
        indices = torch.utils.dlpack.from_dlpack(indices.to_dlpack())[:, 0]
        self.weights[indices] += 1

    #* 不直接从hashmap中提取数据，而是从tensor中直接获取，避免多次数据转换
    def _query_tensor(self, keys):
        """[summary]
        Args:
            keys ([torch.Tensor]): shape: [1, 8, B, N, 3] torch.Size([1, 8, 1000, 35, 3])
        Returns:
            [type]: [description]
        """
        shapes = [s for s in keys.shape]
        n_pts = np.asarray(shapes[:-1]).prod()
        assert shapes[-1] == 3
        #* 初始化输出
        out_feats = torch.zeros([n_pts, self.n_feats], device=self.device)
        out_weights = torch.zeros([n_pts, 1], device=self.device)
        out_num_hits = torch.zeros([n_pts, 1], device=self.device)
        o3c_keys = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(keys.reshape(-1, 3).long()))
        buf_indices, masks = self.tensor_indexer.find(o3c_keys)
        buf_indices = buf_indices[masks]
        indices = self.tensor_indexer.value_tensor()[buf_indices]
        indices = torch.utils.dlpack.from_dlpack(indices.to_dlpack())[:, 0] #* 转为tensor
        masks_torch = torch.utils.dlpack.from_dlpack(masks.to(o3c.int64).to_dlpack()).bool()
        out_feats[masks_torch] = self.features[indices]
        out_weights[masks_torch] = self.weights[indices]
        out_num_hits[masks_torch] = self.num_hits[indices]
        out_feats = out_feats.reshape(shapes[:-1] + [self.n_feats])
        out_weights = out_weights.reshape(shapes[:-1] + [1])
        out_num_hits = out_num_hits.reshape(shapes[:-1] + [1])

        return out_feats, out_weights, out_num_hits

    #* 直接从hashmap中提取数据
    def query(self, keys):
        """[summary]
        Args:
            keys ([torch.Tensor]): shape: [..., 3]
        Returns:
            [type]: [description]
        """
        #* fine_coords torch.Size([45232, 3]) 有效格子坐标
        shapes = [s for s in keys.shape]
        n_pts = np.asarray(shapes[:-1]).prod() #* 计算数组所有元素相乘，有axis限制的话计算沿着轴的元素相乘
        assert shapes[-1] == 3
        if n_pts == 0:
            return None, None, None
        out_feats = torch.zeros((n_pts, self.n_feats), device=self.device)
        out_weights = torch.zeros((n_pts, 1), device=self.device)
        out_num_hits = torch.zeros((n_pts, 1), device=self.device)
        #* DLPack 是张量数据结构的中间内存表示标准，允许张量数据在框架之间交换 请注意，每个dlpack只能使用一次。
        o3c_keys = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(keys.reshape(-1, 3).long()))
        buf_inds, masks = self.indexer.find(o3c_keys)
        buf_inds = buf_inds[masks].to(o3c.int64)
        if not len(buf_inds) == 0:
            masks_torch = torch.utils.dlpack.from_dlpack(masks.to(o3c.int64).to_dlpack()).bool()
            out_feats[masks_torch] = torch.utils.dlpack.from_dlpack(self.indexer.value_tensor(0)[buf_inds].to_dlpack())
            out_weights[masks_torch] = torch.utils.dlpack.from_dlpack(self.indexer.value_tensor(1)[buf_inds].to_dlpack())
            out_num_hits[masks_torch] = torch.utils.dlpack.from_dlpack(self.indexer.value_tensor(2)[buf_inds].to_dlpack())
            out_feats = out_feats.reshape(shapes[:-1] + [self.n_feats]) #* [N, 8]
            out_weights = out_weights.reshape(shapes[:-1] + [1]) #* [N, 1]
            out_num_hits = out_num_hits.reshape(shapes[:-1] + [1]) #* [N, 1]
        return out_feats, out_weights, out_num_hits

    def meshlize(self, nerf, sdf_delta=None, path=None):
        assert self.active_coordinates is not None, "call self.to_tensor() first."
        active_pts = self.active_coordinates * self.voxel_size + self.min_coords
        active_pts = active_pts.detach().cpu().numpy()
        active_coords = self.active_coordinates.detach().cpu().numpy() #* (27500, 3)
        batch_size = 500
        step_size = 0.5
        level = 0.
        all_vertices = []
        all_faces = []
        last_face_id = 0
        for i in range(0, len(self.active_coordinates), batch_size):
            origin = active_coords[i: i + batch_size]
            n_batches = len(origin)
            range_ = np.arange(0, 1+step_size, step_size) - 0.5
            spacing = [range_[1] - range_[0]] * 3
            voxel_coords = np.stack(np.meshgrid(range_, range_, range_, indexing="ij"),axis=-1)
            voxel_coords = np.tile(voxel_coords, (n_batches, 1, 1, 1, 1))
            voxel_coords += origin[:, None, None, None, :]
            voxel_coords = torch.from_numpy(voxel_coords).float().to(self.device)
            H, W, D = voxel_coords.shape[1:4]
            voxel_coords = voxel_coords.reshape(1, n_batches, -1, 3)
            out = self.decode_pts(voxel_coords,nerf,sdf_delta,is_coords=True)
            sdf = out[0, :, :, 0].reshape(n_batches, H, W, D)
            sdf = sdf.detach().cpu().numpy() #* (500, 3, 3, 3)
            # import pdb;pdb.set_trace()
            for j in range(n_batches):
                if np.max(sdf[j]) > level and np.min(sdf[j]) < level:
                    verts, faces, normals, values = marching_cubes(sdf[j], level=level, spacing=spacing)
                    verts += origin[j] - 0.5
                    all_vertices.append(verts)
                    all_faces.append(faces + last_face_id)
                    last_face_id += np.max(faces) + 1
                    import pdb;pdb.set_trace()
        if len(all_vertices) == 0:
            return None
        final_vertices = np.concatenate(all_vertices, axis=0)
        final_faces = np.concatenate(all_faces, axis=0)
        final_vertices = final_vertices * self.voxel_size + self.min_coords.cpu().numpy()
        # all_normals = np.concatenate(all_normals, axis=0)
        mesh = trimesh.Trimesh(vertices=final_vertices, faces=final_faces, process=False)
        if path is not None:
            mesh.export(path)
        return active_pts, mesh

    def decode_pts(
        self,
        coords, #* 网格下的坐标(浮点数)
        nerf,
        sdf_delta=None,
        is_coords=False,
        query_tensor=True,
    ):
        """ decode sdf values from the implicit volume given coords.
        Args:
            coords (_type_): [1, 8, n_pts, 3], input pts.
            nerf (_type_): _description_
            voxel_size (_type_): _description_
            sdf_delta (_type_, optional): _description_. Defaults to None.
            is_coords (_type_, optional): True if input pts are in voxel coords.
                Otherwise, they are in world coordinate that should be converted
                to voxel coords first.
            volume_resolution (_type_, optional): _description_. Defaults to None.
        Returns:
            _type_: _description_
        """

        if not is_coords:
            coords = (coords - self.min_coords) / self.voxel_size
        #* 每个采样点周围的8个网格
        neighbor_coords = get_neighbors(coords) #* torch.Size([1, 8, 1000, 35, 3])
        #* 每个采样点到周围网格的相对位移，以网格点为原点
        local_coords = coords.unsqueeze(1) - neighbor_coords #* torch.Size([1, 8, 1000, 35, 3])
        assert torch.min(local_coords) >= -1
        assert torch.max(local_coords) <= 1
        # import pdb;pdb.set_trace()
        weights_unmasked = torch.prod(1 - torch.abs(local_coords),dim=-1,keepdim=True) #* torch.Size([1, 8, 1000, 35, 1])
        #* 距离点越近，权重越大
        #* 获取网格坐标的特征和权重
        if query_tensor:
            feats, weights, num_hits = self._query_tensor(neighbor_coords)
        else:
            feats, weights, num_hits = self.query(neighbor_coords)
        #* feats.shape torch.Size([1, 8, 1000, 35, 8])
        #*  weights.shape torch.Size([1, 8, 1000, 35, 1])
        mask = torch.min(weights, dim=1)[0] > self.min_pts_in_grid
        local_coords_encoded = nerf.xyz_encoding(local_coords) #* 位置编码：torch.Size([1, 8, 1000, 35, 9]) 3维变9维
        # local_coords[0,0,0,0,:] tensor([0.9791, 0.1294, 0.0445], device='cuda:0')
        # torch.sin(local_coords[0,0,0,0,:]) tensor([0.8300, 0.1290, 0.0445], device='cuda:0')
        # torch.cos(local_coords[0,0,0,0,:]) tensor([0.5578, 0.9916, 0.9990], device='cuda:0')
        # local_coords_encoded[0,0,0,0,:] tensor([0.9791, 0.1294, 0.0445, 0.8300, 0.1290, 0.0445, 0.5578, 0.9916, 0.9990], device='cuda:0')
        #* 将点相对位姿及其编码和特征拼接，得到17维向量
        nerf_in = torch.cat([local_coords_encoded, feats], dim=-1) #* torch.Size([1, 8, 1000, 35, 17])
        #* 多层MLP得到sdf，解码的时候用谁的特征就以谁为原点
        alpha = nerf.geo_forward(nerf_in) #* torch.Size([1, 8, 1000, 35, 1])
        alpha = alpha * self.voxel_size #* 0.01 ？？
        normalizer = torch.sum(weights_unmasked, dim=1, keepdim=True)
        #* 权重归一化
        weights_unmasked = weights_unmasked / normalizer
        assert torch.all(torch.abs(weights_unmasked.sum(1) - 1) < 1e-5)
        alpha = torch.sum(alpha * weights_unmasked, dim=1) #* 对于周围的8点加权求和 torch.Size([1, 1000, 35, 1])
        if sdf_delta is not None: #* torch.Size([1, 1, 240, 108, 122])
            neighbor_coords_grid_sample = neighbor_coords / (self.n_xyz-1) #* torch.Size([1, 8, 1000, 35, 3])
            neighbor_coords_grid_sample = neighbor_coords_grid_sample * 2 - 1
            neighbor_coords_grid_sample = neighbor_coords_grid_sample[..., [2, 1, 0]]
            #* grid[n, d, h, w]
            sdf_delta = F.grid_sample(
                sdf_delta,
                neighbor_coords_grid_sample,  # [1, 8, n_pts, n_steps, 3]
                mode="nearest",
                padding_mode="zeros",
                align_corners=True
            )
            sdf_delta = sdf_delta.permute(0, 2, 3, 4, 1)  # [B, 8, N, S, 1]
            sdf_delta = torch.sum(sdf_delta * weights_unmasked, dim=1)
            alpha += sdf_delta
        return alpha

    def save(self, path):
        self.print_statistic()
        active_buf_indices = self.tensor_indexer.active_buf_indices().to(o3c.int64)

        active_keys = self.tensor_indexer.key_tensor()[active_buf_indices]
        active_keys = torch.utils.dlpack.from_dlpack(active_keys.to_dlpack())
        
        active_vals = self.tensor_indexer.value_tensor()[active_buf_indices]
        active_vals = torch.utils.dlpack.from_dlpack(active_vals.to_dlpack())

        out_dict = {
            "25%": self.per_25 if self.per_25 else None,
            "50%": self.per_50,
            "75%": self.per_75,
            "dimensions": self.dimensions,
            "voxel_size": self.voxel_size,
            "mean": self.avg_n_pts,
            "min": self.min_pts,
            "active_keys": active_keys,
            "active_vals": active_vals,
            "features": self.features,
            "weights": self.weights,
            "num_hits": self.num_hits,
            "active_coordinates": self.active_coordinates
        }
        torch.save(out_dict, path + "_sparse_volume.pth")

    def load(self, path):
        volume = torch.load(path)
        active_keys = volume['active_keys']
        active_vals = volume['active_vals']
        features = volume['features']
        weights = volume['weights']
        num_hits = volume['num_hits']
        active_coordinates = volume['active_coordinates']

        self.tensor_indexer = o3c.HashMap(
            len(active_keys),
            key_dtype=o3c.int64,
            key_element_shape=(3,),
            value_dtype=o3c.int64,
            value_element_shape=(1,),
            device=o3c.Device(self.device)
        )
        active_keys = o3c.Tensor.from_dlpack(
            torch.utils.dlpack.to_dlpack(active_keys))
        active_vals = o3c.Tensor.from_dlpack(
            torch.utils.dlpack.to_dlpack(active_vals))

        buf_indices, masks = self.tensor_indexer.insert(
            active_keys, active_vals)
        masks = masks.cpu().numpy() if "cuda" in self.device else masks.numpy()
        assert masks.all()

        self.active_coordinates = active_coordinates
        self.features = features
        self.weights = weights
        self.num_hits = num_hits

    def print_statistic(self):
        print("===========")
        p = np.percentile(self.n_pts_list, [25, 50, 75])
        self.per_25 = p[0]
        self.per_50 = p[1]
        self.per_75 = p[2]
        print(f"25%: {p[0]}, 50%: {p[1]}, 75%:{p[2]}")
        print(f"mean: {self.avg_n_pts}, min: {self.min_pts}, max:{self.max_pts}")
        print("===========")
    
    def track_n_pts(self, n_pts):
        self.n_pts_list.append(float(n_pts))
        self.avg_n_pts = (self.avg_n_pts * self.n_frames + n_pts) / (self.n_frames + 1)
        self.n_frames += 1
        self.min_pts = min(self.min_pts, n_pts)
        self.max_pts = max(self.max_pts, n_pts)


if __name__ == "__main__":
    o3c_device = o3c.Device("cuda:0")
    volume = SparseVolume()
    import pdb;pdb.set_trace()


