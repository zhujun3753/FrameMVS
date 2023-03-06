import os
os.environ['NUMEXPR_MAX_THREADS'] = "24"
import numpy as np
import open3d as o3d
import open3d.core as o3c
import trimesh
import torch
from typing import Union, Tuple, Sequence
from matplotlib import pyplot as plt



def get_homogeneous(
    pts: Union['np.ndarray', 'torch.tensor']
    ) -> Union['np.ndarray', 'torch.tensor']:
    """ convert [(b), N, 3] pts to homogeneous coordinate

    Args:
        pts ([(b), N, 3] Union['np.ndarray', 'torch.tensor']): input point cloud

    Returns:
        homo_pts ([(b), N, 4] Union['np.ndarray', 'torch.tensor']): output point
            cloud

    Raises:
        ValueError: if the input tensor/array is not with the shape of [b, N, 3]
            or [N, 3]
        TypeError: if input is not either tensor or array
    """

    batch = False
    if len(pts.shape) == 3:
        batch = True
    elif len(pts.shape) == 2:
        pts = pts
    else:
        raise ValueError("only accept [b, n_pts, 3] or [n_pts, 3]")

    if isinstance(pts, torch.Tensor):
        ones = torch.ones_like(pts[..., :1])
        homo_pts = torch.cat([pts, ones], dim=-1)
    elif isinstance(pts, np.ndarray):
        ones = np.ones_like(pts[..., :1])
        homo_pts = np.concatenate([pts, ones], axis=-1)
    else:
        raise TypeError("wrong data type")
    return homo_pts

def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    if np.sum(b + a) == 0:  # if b is possite to a
        b += 1e-3
    axis_ = np.cross(a, b)
    axis_ = axis_ / (np.linalg.norm(axis_))
    angle = np.arccos(np.dot(a, b))
    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2

def load_scene_mesh(path, trans_mat=None, open_3d=True):
    scene_mesh = trimesh.load(path)
    if trans_mat is not None:
        scene_mesh.vertices = np.dot(get_homogeneous(scene_mesh.vertices), trans_mat.T)[:, :3]
    if open_3d:
        scene_mesh_o3d = trimesh2o3d(scene_mesh)
        return scene_mesh_o3d
    else:
        return scene_mesh


def trimesh2o3d(mesh):
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces)
    mesh_o3d.compute_vertex_normals()
    if mesh.visual.vertex_colors is not None:
        mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(
            mesh.visual.vertex_colors[:, :3] / 255.
        )
    return mesh_o3d


def np2pc(points, colors=None):
    """ convert numpy colors point cloud to o3d point cloud

    Args:
        points (np.ndarray): [n_pts, 3]
        colors (np.ndarray): [n_pts, 3]
    Return:
        pts_o3d (o3d.geometry.PointCloud)
    """
    pts_o3d = o3d.geometry.PointCloud()
    pts_o3d.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pts_o3d.colors = o3d.utility.Vector3dVector(colors)
    return pts_o3d


def mesh2o3d(vertices, faces, normals=None, colors=None):
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=normals,
        vertex_colors=colors
    )
    return trimesh2o3d(mesh)


def post_process_mesh(mesh, vertex_threshold=0.005, surface_threshold=0.1):
    """ merge close vertices and remove small connected components
    Args:
        mesh (trimesh.Trimesh): input trimesh

    Returns:
        _type_: _description_
    """
    mesh_o3d = trimesh2o3d(mesh)
    mesh_o3d.merge_close_vertices(vertex_threshold).remove_degenerate_triangles().remove_duplicated_triangles().remove_duplicated_vertices()
    mesh_o3d = mesh_o3d.filter_smooth_simple(number_of_iterations=1)
    # component_ids, component_nums, component_surfaces = mesh_o3d.cluster_connected_triangles()
    # remove_componenets = np.asarray(component_nums)[np.asarray(component_surfaces) < surface_threshold]
    # remove_mask = [c in remove_componenets for c in component_ids]
    # mesh_o3d.remove_triangles_by_mask(remove_mask)
    # mesh_o3d.remove_unreferenced_vertices()
    mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh_o3d.vertices),
        faces=np.asarray(mesh_o3d.triangles),
    )
    return mesh

class TSDFFusion:
    def __init__(self, voxel_length=0.01, sdf_trunc=0.02):
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_length,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    
    def integrate(self, depth, color, T_wc, intr_mat):
        """integrate new RGBD frame
        Args:
            depth (np.ndarray): [h,w] in range[0,255]
            color (np.ndarray): [h,w,3] in meters
            T_wc (np.ndarray): [4,4]
            intr_mat (np.ndarray): [3,3] or [4,4]
        """
        img_h, img_w = depth.shape
        # import pdb;pdb.set_trace() #* np.asarray(color, order="C") RuntimeError: Image can only be initialized from c-style buffer.
        color = o3d.geometry.Image(np.asarray(color, order="C").astype(np.uint8))
        depth = o3d.geometry.Image((depth * 1000).astype(np.uint16))
        #* depth_scale (float, optional, default=1000.0) – The ratio to scale depth values. The depth values will first be scaled and then truncated.
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, depth_trunc=40.0, convert_rgb_to_intensity=False)
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(
            width=img_w,
            height=img_h,
            fx=intr_mat[0, 0],
            fy=intr_mat[1, 1],
            cx=intr_mat[0, 2],
            cy=intr_mat[1, 2],
        )
        T_cw = np.linalg.inv(T_wc)
        self.volume.integrate(rgbd, intrinsic, T_cw)
    
    def marching_cube(self, path=None):
        # print("Extract a triangle mesh from the volume and visualize it.")
        mesh_o3d = self.volume.extract_triangle_mesh()
        mesh_o3d.compute_vertex_normals()
        
        if path is not None:
            mesh_out = mesh_o3d.filter_smooth_simple(number_of_iterations=2)
            mesh_out.compute_vertex_normals()
            o3d.io.write_triangle_mesh(path,mesh_out)
        # import pdb;pdb.set_trace()
        # o3d.visualization.draw_geometries([mesh_o3d], front=[0.5297, -0.1873, -0.8272],
        #                                     lookat=[2.0712, 2.0312, 1.7251],
        #                                     up=[-0.0558, -0.9809, 0.1864], zoom=0.47)
        # mesh = trimesh.Trimesh(
        #     vertices=np.asarray(mesh_o3d.vertices), # / dimension,
        #     faces=np.asarray(mesh_o3d.triangles),
        #     vertex_normals=np.asarray(mesh_o3d.vertex_normals)
        # )
        # if path is not None:
        #     dir_ = "/".join(path.split("/")[:-1])
        #     if not os.path.exists(dir_):
        #         os.mkdir(dir_)
        #     mesh.export(path)
        return mesh_o3d


if __name__ == "__main__":
    mesh = trimesh.load("/home/kejie/repository/bnv_fusion/logs/run_e2e/scene0000_00/3999.ply")
    mesh = post_process_mesh(mesh)
