import cv2
import imageio
import numpy as np
import os
import os.path as osp
import torch
import time
import skimage
from skimage import transform


import torch.distributed as dist

def make_nograd_func(func):
    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            ret = func(*f_args, **f_kwargs)
        return ret

    return wrapper


# convert a function into recursive style to handle nested dict/list/tuple variables
def make_recursive_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper


@make_recursive_func
def tensor2float(vars):
    if isinstance(vars, float):
        return vars
    elif isinstance(vars, torch.Tensor):
        if len(vars.shape) == 0:
            return vars.data.item()
        else:
            return [v.data.item() for v in vars]
    else:
        raise NotImplementedError("invalid input type {} for tensor2float".format(type(vars)))


@make_recursive_func
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.detach().cpu().numpy().copy()
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


@make_recursive_func
def tocuda(vars):
    if isinstance(vars, torch.Tensor):
        return vars.cuda()
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


def save_scalars(logger, mode, scalar_dict, global_step):
    scalar_dict = tensor2float(scalar_dict)
    for key, value in scalar_dict.items():
        if not isinstance(value, (list, tuple)):
            name = '{}/{}'.format(mode, key)
            logger.add_scalar(name, value, global_step)
        else:
            for idx in range(len(value)):
                name = '{}/{}_{}'.format(mode, key, idx)
                logger.add_scalar(name, value[idx], global_step)


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

class Timer:
    def __init__(self, names):
        self.times = {n: 0 for n in names}
        self.t0 = {n: 0 for n in names}

    def start(self, name):
        self.t0[name] = time.time()
    
    def log(self, name):
        self.times[name] += time.time() - self.t0[name]


def to_cuda(in_dict):
    for k in in_dict:
        if isinstance(in_dict[k], torch.Tensor):
            in_dict[k] = in_dict[k].to("cuda")


def to_cpu(in_dict):
    for k in in_dict:
        if isinstance(in_dict[k], torch.Tensor):
            in_dict[k] = in_dict[k].cpu()


def override_weights(model, pretrained_weights, keys):
    """
    Args:
        model: pytorch nn module
        pretrained_weights: OrderedDict of state_dict
        keys: a list of keyword. the weights to be overrided if matched 
    """

    pretrained_dict = {}
    for model_key in model.state_dict().keys():
        if any([(key in model_key) for key in keys]):
            if model_key not in pretrained_weights:
                print(f"[warning]: {model_key} not in pretrained weight")
                continue
            pretrained_dict[model_key] = pretrained_weights[model_key] 
    model.load_state_dict(pretrained_dict, strict=False)


def get_file_paths(dir, file_type=None):
    names = sorted(os.listdir(dir))
    out = []
    for n in names:
        if os.path.isdir(osp.join(dir, n)):
            paths = get_file_paths(osp.join(dir, n), file_type)
            out.extend(paths)
        else:
            if file_type is not None:
                if n.endswith(file_type):
                    out.append(osp.join(dir, n))
            else:
                out.append(osp.join(dir, n))
    return out


def inverse_sigmoid(x):
    return np.log(x) - np.log(1-x)


def load_rgb(path, downsample_scale=0):
    img = imageio.imread(path)
    img = skimage.img_as_float32(img)
    # print(img.max(),img.min())

    if downsample_scale > 0:
        img = transform.rescale(img, (downsample_scale, downsample_scale, 1))
    # pixel values between [-1,1]
    # print(img.max(),img.min())
    img -= 0.5
    img *= 2.
    img = img.transpose(2, 0, 1)
    return img


def load_depth(
    path,
    downsample_scale,
    downsample_mode="dense",
    max_depth=None,
    add_noise=False
):
    depth = cv2.imread(path, -1) / 1000.
    if downsample_scale > 0:
        img_h, img_w = depth.shape
        if downsample_mode == "dense":
            reduced_w = int(img_w * downsample_scale)
            reduced_h = int(img_h * downsample_scale)
            depth = cv2.resize(depth,dsize=(reduced_w, reduced_h),interpolation=cv2.INTER_NEAREST)
        else:
            assert downsample_mode == "sparse"
            downsample_mask = np.zeros_like(depth)
            interval = int(1 / downsample_scale)
            downsample_mask[::interval, ::interval] = 1
            depth = depth * downsample_mask
    mask = depth > 0
    if max_depth is not None:
        mask *= depth < max_depth
        depth = depth * mask
    if add_noise:
        noise_depth = noise_simulator.simulate(depth)
        # noise_depth = add_depth_noise(depth)
        noise_depth = noise_depth * mask
        return depth, noise_depth, mask
    else:
        return depth, depth, mask
