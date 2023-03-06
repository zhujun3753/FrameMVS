import sys
sys.path.append('.')

import time
from tools.tsdf_fusion.fusion import *
import pickle
import argparse
from tqdm import tqdm
import ray
import torch.multiprocessing
from tools.simple_loader import *
from PIL import Image
import trimesh
# from tools.evaluation import Renderer
import pyrender

torch.multiprocessing.set_sharing_strategy('file_system')

viewer_flags = {
    "lighting_intensity": 4.0,
    "window_title": "NeuralRecon reconstructions",
}
class Renderer:
    def __init__(self,mesh,mesh_pose,height=480, width=640):
        # pass
        # self.renderer = pyrender.OffscreenRenderer(width, height)
        # # self.scene = pyrender.Scene(ambient_light=[0,0,0, 0.], bg_color=[1.0, 1.0, 1.0, 1])
        self.scene = pyrender.Scene()

        
        # self.mesh_node = None
        mesh = pyrender.Mesh.from_trimesh(mesh)
        # # self.viewer.render_lock.acquire()
        self.scene.add(mesh)
        self.cam_node=None
        # if self.mesh_node is not None:
        #     self.scene.remove_node(self.mesh_node)
        # self.viewer.render_lock.release()
        # self.mesh_node = list(self.scene.mesh_nodes)[0]
        # import pdb;pdb.set_trace()
    
    def __call__(self, height, width, intrinsics, pose, mesh):
        # mesh = pyrender.Mesh.from_trimesh(mesh)
        # scene = pyrender.Scene()
        # scene.add(mesh)
        cam = pyrender.IntrinsicsCamera(cx=intrinsics[0, 2], cy=intrinsics[1, 2],
                                        fx=intrinsics[0, 0], fy=intrinsics[1, 1])
        if self.cam_node is not None:
            self.scene.remove_node(self.cam_node)
        self.cam_node=self.scene.add(cam, pose=self.fix_pose(pose))
        # scene.add(point_l, pose=self.fix_pose(pose))
        # v = pyrender.Viewer(scene)
        r = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
        color, depth = r.render(self.scene,flags=pyrender.constants.RenderFlags.FLAT)
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(color)
        # plt.show()
        return color, depth
        # import pdb;pdb.set_trace()
    
    def fix_pose(self, pose):
        # 3D Rotation about the x-axis.
        t = np.pi
        c = np.cos(t)
        s = np.sin(t)
        R = np.array([[1, 0, 0],
                      [0, c, -s],
                      [0, s, c]])
        axis_transform = np.eye(4)
        axis_transform[:3, :3] = R
        return pose @ axis_transform

    def vis_mesh(self, mesh: trimesh.Trimesh):
        mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        self.viewer.render_lock.acquire()
        self.scene.add(mesh)
        if self.mesh_node is not None:
            self.scene.remove_node(self.mesh_node)
        self.viewer.render_lock.release()
        self.mesh_node = list(self.scene.mesh_nodes)[0]
        # self.count+=1
        # if self.count==55:
        #     import pdb;pdb.set_trace()

    def close(self):
        self.viewer.close_external()
    
    def delete(self):
        self.renderer.delete()

def parse_args():
    parser = argparse.ArgumentParser(description='Fuse ground truth tsdf')
    parser.add_argument("--dataset", default='scannet')
    parser.add_argument("--data_path", metavar="DIR",help="path to raw dataset", default='/data/scannet/output/')
    parser.add_argument("--save_name", metavar="DIR",help="file name", default='all_tsdf')
    parser.add_argument('--test', action='store_true',help='prepare the test set')
    parser.add_argument('--max_depth', default=3., type=float,help='mask out large depth values since they are noisy')
    parser.add_argument('--num_layers', default=3, type=int)
    parser.add_argument('--margin', default=3, type=int)
    parser.add_argument('--voxel_size', default=0.01, type=float) #* 0.04
    parser.add_argument('--window_size', default=9, type=int)
    parser.add_argument('--min_angle', default=15, type=float)
    parser.add_argument('--min_distance', default=0.1, type=float)
    # ray multi processes
    parser.add_argument('--n_proc', type=int, default=1, help='#processes launched to process scenes.')
    parser.add_argument('--n_gpu', type=int, default=1, help='#number of gpus')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--loader_num_workers', type=int, default=4)
    return parser.parse_args()

args = parse_args()
args.save_path = os.path.join(args.data_path, args.save_name)

def read_img( filepath):
        img = Image.open(filepath)
        return img

def read_depth( filepath):
    # Read depth image and camera pose
    depth_im = cv2.imread(filepath, -1).astype(np.float32)
    depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
    depth_im[depth_im > 4.0] = 0
    return depth_im

def tsdf_fusion(datapath,save_mesh=True,
                save_path='/home/zhujun/MVS/ARKitScenes-main/data/3dod/Training/all_tsdf_9/40753679'):
    vol_bnds = np.zeros((3, 2))
    with open(os.path.join(datapath, 'fragments.pkl'), 'rb') as f:
        metas = pickle.load(f)
    for meta in metas:
        imgs = []
        depth=[]
        intrinsics_list = meta['intrinsics']
        extrinsics_list = meta['extrinsics']
        for i, vid in enumerate(meta['image_ids']):
            imgs.append(read_img(os.path.join(datapath, 'images', '{}.jpg'.format(vid))))
            depth.append(read_depth(os.path.join(datapath, 'depth', '{}.png'.format(vid))))
        for i in range(len(intrinsics_list)):
            depth_im = depth[i]
            cam_pose = extrinsics_list[i]
            cam_intr=intrinsics_list[i]
            # Compute camera view frustum and extend convex hull
            view_frust_pts = get_view_frustum(depth_im, cam_intr, cam_pose) #* 获得当前图片的全局坐标系视锥的五个顶点（3，5）
            vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
            vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
    # Integrate
    # Initialize voxel volume
    print("Initializing voxel volume...")
    tsdf_vol_list = []
    for l in range(3):
        tsdf_vol_list.append(TSDFVolume(vol_bnds, voxel_size=0.01 * 2 ** l, margin=3))
    # Loop through RGB-D images and fuse them together
    for meta in metas:
        imgs = []
        depth=[]
        intrinsics_list = meta['intrinsics']
        extrinsics_list = meta['extrinsics']
        for i, vid in enumerate(meta['image_ids']):
            imgs.append(read_img(os.path.join(datapath, 'images', '{}.jpg'.format(vid))))
            depth.append(read_depth(os.path.join(datapath, 'depth', '{}.png'.format(vid))))
        for i in range(len(intrinsics_list)):
            depth_im = depth[i]
            cam_pose = extrinsics_list[i]
            cam_intr=intrinsics_list[i]
            color_image = imgs[i]
            # Integrate observation into voxel volume (assume color aligned with depth)
            for l in range(3):
                tsdf_vol_list[l].integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)
    
    if save_mesh:
        for l in range(3):
            print("Saving mesh to mesh{}.ply...".format(str(l)))
            verts, faces, norms, colors = tsdf_vol_list[l].get_mesh()
            meshwrite(os.path.join(save_path,  'mesh_layer{}.ply'.format(str(l))), verts, faces, norms, colors)
            # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
            print("Saving point cloud to pc.ply...",os.path.join(save_path,  'pc_layer{}.ply'.format(str(l))))
            point_cloud = tsdf_vol_list[l].get_point_cloud()
            pcwrite(os.path.join(save_path,  'pc_layer{}.ply'.format(str(l))), point_cloud)

def save_tsdf_full(args, scene_path, cam_intr, depth_list, cam_pose_list, color_list, save_mesh=False):
    # ======================================================================================================== #
    # (Optional) This is an example of how to compute the 3D bounds
    # in world coordinates of the convex hull of all camera view
    # frustums in the dataset
    # ======================================================================================================== #
    vol_bnds = np.zeros((3, 2))
    n_imgs = len(depth_list.keys())
    if n_imgs > 600:
        ind = np.linspace(0, n_imgs - 1, 200).astype(np.int32)
        image_id = np.array(list(depth_list.keys()))[ind]
    else:
        image_id = depth_list.keys()
    #* 确定每个坐标轴方向的最小值和最大值来作为体素网格边界
    for i,id in enumerate(image_id):
        depth_im = depth_list[id]
        cam_pose = cam_pose_list[id]
        # Compute camera view frustum and extend convex hull
        view_frust_pts = get_view_frustum(depth_im, cam_intr, cam_pose) #* 获得当前图片的全局坐标系视锥的五个顶点（3，5）
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
    # Integrate
    # Initialize voxel volume
    print("Initializing voxel volume...")
    tsdf_vol_list = []
    for l in range(args.num_layers):
        tsdf_vol_list.append(TSDFVolume(vol_bnds, voxel_size=args.voxel_size * 2 ** l, margin=args.margin))
    # Loop through RGB-D images and fuse them together
    t0_elapse = time.time()
    for id in depth_list.keys():
        # if id<50:
        #     continue
        if id % 100 == 0:
            print("{}: Fusing frame {}/{}".format(scene_path, str(id), str(n_imgs)))
        depth_im = depth_list[id]
        cam_pose = cam_pose_list[id]
        if len(color_list) == 0:
            color_image = None
        else:
            color_image = color_list[id]
        # Integrate observation into voxel volume (assume color aligned with depth)
        for l in range(args.num_layers):
            tsdf_vol_list[l].integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)
    fps = n_imgs / (time.time() - t0_elapse)
    print("Average FPS: {:.2f}".format(fps))
    tsdf_info = {
        'vol_origin': tsdf_vol_list[0]._vol_origin,
        'voxel_size': tsdf_vol_list[0]._voxel_size,
    }
    tsdf_path = os.path.join(args.save_path, scene_path)
    if not os.path.exists(tsdf_path):
        os.makedirs(tsdf_path)
    with open(os.path.join(args.save_path, scene_path, 'tsdf_info.pkl'), 'wb') as f:
        pickle.dump(tsdf_info, f)
    for l in range(args.num_layers):
        tsdf_vol, color_vol, weight_vol = tsdf_vol_list[l].get_volume()
        np.savez_compressed(os.path.join(args.save_path, scene_path, 'full_tsdf_layer{}'.format(str(l))), tsdf_vol)
    if save_mesh:
        for l in range(args.num_layers):
            print("Saving mesh to mesh{}.ply...".format(str(l)))
            verts, faces, norms, colors = tsdf_vol_list[l].get_mesh()
            meshwrite(os.path.join(args.save_path, scene_path, 'mesh_layer{}.ply'.format(str(l))), verts, faces, norms, colors)
            # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
            print("Saving point cloud to pc.ply...",os.path.join(args.save_path, scene_path, 'pc_layer{}.ply'.format(str(l))))
            point_cloud = tsdf_vol_list[l].get_point_cloud()
            pcwrite(os.path.join(args.save_path, scene_path, 'pc_layer{}.ply'.format(str(l))), point_cloud)

def render_depth_color(args, scene_path, cam_intr, depth_list, cam_pose_list, color_list):
    l=1
    mesh_file = os.path.join(args.save_path, scene_path, 'mesh_layer{}.ply'.format(str(l)))
    mesh = trimesh.load(mesh_file)
    mesh_pose=np.eye(4)
    # mesh renderer
    height, width=depth_list[0].shape[:2]
    renderer = Renderer(mesh,mesh_pose,height, width)
    
    n_imgs = len(depth_list.keys())
    for id in depth_list.keys():
        if id % 100 == 0:
            print("{}: renderer frame {}/{}".format(scene_path, str(id), str(n_imgs)))
        depth_im = depth_list[id]
        cam_pose = cam_pose_list[id]
        if len(color_list) == 0:
            color_image = None
        else:
            color_image = color_list[id]
        color_pred, depth_pred = renderer(height, width, cam_intr, cam_pose, mesh)
        Image.fromarray(color_pred).save(os.path.join(args.save_path, scene_path, 'render_images',str(id).zfill(5)+'.jpg'))
        cv2.imwrite(os.path.join(args.save_path, scene_path, 'render_depth',str(id).zfill(5)+'.png'), (depth_pred*1000).astype(np.uint16) )
        # Image.fromarray(color_image).save(os.path.join(args.save_path, scene_path, 'color_image{}.png'.format(str(l))))

    import pdb;pdb.set_trace()
        


def save_fragment_pkl(args, scene, cam_intr, depth_list, cam_pose_list):
    fragments = []
    print('segment: process scene {}'.format(scene))
    # gather pose
    vol_bnds = np.zeros((3, 2))
    vol_bnds[:, 0] = np.inf
    vol_bnds[:, 1] = -np.inf
    all_ids = []
    ids = []
    all_bnds = []
    count = 0
    last_pose = None
    for id in depth_list.keys():
        depth_im = depth_list[id]
        cam_pose = cam_pose_list[id]
        if count == 0:
            ids.append(id)
            vol_bnds = np.zeros((3, 2))
            vol_bnds[:, 0] = np.inf
            vol_bnds[:, 1] = -np.inf
            last_pose = cam_pose
            # Compute camera view frustum and extend convex hull
            view_frust_pts = get_view_frustum(depth_im, cam_intr, cam_pose)
            vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
            vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
            count += 1
        else:
            angle = np.arccos(((np.linalg.inv(cam_pose[:3, :3]) @ last_pose[:3, :3] @ np.array([0, 0, 1]).T) * np.array([0, 0, 1])).sum())
            dis = np.linalg.norm(cam_pose[:3, 3] - last_pose[:3, 3])
            if angle > (args.min_angle / 180) * np.pi or dis > args.min_distance:
                ids.append(id)
                last_pose = cam_pose
                # Compute camera view frustum and extend convex hull
                view_frust_pts = get_view_frustum(depth_im, cam_intr, cam_pose)
                vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
                vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
                count += 1
                if count == args.window_size:
                    all_ids.append(ids)
                    all_bnds.append(vol_bnds)
                    ids = []
                    count = 0
    with open(os.path.join(args.save_path, scene, 'tsdf_info.pkl'), 'rb') as f:
        tsdf_info = pickle.load(f)
    # save fragments
    for i, bnds in enumerate(all_bnds):
        if not os.path.exists(os.path.join(args.save_path, scene, 'fragments', str(i))):
            os.makedirs(os.path.join(args.save_path, scene, 'fragments', str(i)))
        fragments.append({
            'scene': scene,
            'fragment_id': i,
            'image_ids': all_ids[i],
            'vol_origin': tsdf_info['vol_origin'],
            'voxel_size': tsdf_info['voxel_size'],
        })
    # import pdb;pdb.set_trace()
    with open(os.path.join(args.save_path, scene, 'fragments.pkl'), 'wb') as f:
        pickle.dump(fragments, f)

@ray.remote(num_cpus=args.num_workers + 1, num_gpus=(1 / args.n_proc))
def process_with_single_worker(args, scannet_files):
    for scene in tqdm(scannet_files):
        # if os.path.exists(os.path.join(args.save_path, scene, 'fragments.pkl')):
            # continue
        print('read from disk')
        depth_all = {}
        cam_pose_all = {}
        color_all = {}
        if args.dataset == 'scannet':
            n_imgs = len(os.listdir(os.path.join(args.data_path, scene, 'color')))
            intrinsic_dir = os.path.join(args.data_path, scene, 'intrinsic', 'intrinsic_depth.txt')
            cam_intr = np.loadtxt(intrinsic_dir, delimiter=' ')[:3, :3]
            dataset = ScanNetDataset(n_imgs, scene, args.data_path, args.max_depth)
        if args.dataset == 'arkitscene':
            n_imgs = len(os.listdir(os.path.join(args.data_path, scene, scene+'_frames', 'images')))
            intrinsic_dir = os.path.join(args.data_path, scene, scene+'_frames', 'intrinsics', '00000.txt')
            cam_intr = np.loadtxt(intrinsic_dir, delimiter=' ')[:3, :3]
            dataset = ARkitsceneDataset(n_imgs, scene, args.data_path, args.max_depth)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, collate_fn=collate_fn, batch_sampler=None, num_workers=args.loader_num_workers)
        for id, (cam_pose, depth_im, color_image) in enumerate(dataloader):
            if id % 100 == 0:
                print("{}: read frame {}/{}".format(scene, str(id), str(n_imgs)))
            if cam_pose[0][0] == np.inf or cam_pose[0][0] == -np.inf or cam_pose[0][0] == np.nan:
                continue
            depth_all.update({id: depth_im})
            cam_pose_all.update({id: cam_pose})
            color_all.update({id: color_image})
        render_depth_color(args, scene, cam_intr, depth_all, cam_pose_all, color_all)
        # save_tsdf_full(args, scene, cam_intr, depth_all, cam_pose_all, color_all, save_mesh=True)
        import pdb;pdb.set_trace()
        save_fragment_pkl(args, scene, cam_intr, depth_all, cam_pose_all)

def split_list(_list, n):
    assert len(_list) >= n
    ret = [[] for _ in range(n)]
    for idx, item in enumerate(_list):
        ret[idx % n].append(item)
    return ret

def generate_pkl(args):
    all_scenes = sorted(os.listdir(args.save_path))
    # todo: fix for both train/val/test
    if not args.test:
        splits = ['train', 'val']
    else:
        splits = ['test']
    for split in splits:
        fragments = []
        with open(os.path.join(args.save_path, 'splits', 'scannetv2_{}.txt'.format(split))) as f:
            split_files = f.readlines()
        for scene in all_scenes:
            if 'scene' not in scene:
                continue
            # if scene + '\n' in split_files:
            if scene in split_files:
                with open(os.path.join(args.save_path, scene, 'fragments.pkl'), 'rb') as f:
                    frag_scene = pickle.load(f)
                fragments.extend(frag_scene)
        # import pdb;pdb.set_trace()
        # print(fragments)
        # print(all_scenes)
        # print(split_files)
        with open(os.path.join(args.save_path, 'fragments_{}.pkl'.format(split)), 'wb') as f:
            pickle.dump(fragments, f)

if __name__ == "__main__":
    # args = parse_args()
    # args.save_path = os.path.join(args.data_path, args.save_name)
    all_proc = args.n_proc * args.n_gpu
    ray.init(num_cpus=all_proc * (args.num_workers + 1), num_gpus=args.n_gpu)
    if args.dataset == 'scannet':
        if not args.test:
            args.data_path = os.path.join(args.data_path, 'scans')
        else:
            args.data_path = os.path.join(args.data_path, 'scans_test')
        files = sorted(os.listdir(args.data_path))
    elif args.dataset == 'arkitscene':
        # if not args.test:
        #     args.data_path = os.path.join(args.data_path, 'scans')
        # else:
        #     args.data_path = os.path.join(args.data_path, 'scans_test')
        files = sorted(os.listdir(args.data_path))
    else:
        raise NameError('error!')
    # import pdb;pdb.set_trace()
    files = split_list(files, all_proc)
    ray_worker_ids = []
    for w_idx in range(all_proc):
        ray_worker_ids.append(process_with_single_worker.remote(args, files[w_idx]))
    results = ray.get(ray_worker_ids)
    print('generate_pkl')
    if args.dataset == 'scannet':
        generate_pkl(args)
