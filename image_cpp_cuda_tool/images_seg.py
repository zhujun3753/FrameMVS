import os
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
# from images_seg.myimage_seg_cuda_batch.python_code.ball_query_example import region_seg_cuda
# from model import DexiNed
from image_cpp_cuda_tool.model import DexiNed

from my_image_cpp_cuda_tool import my_image_tool
from time import time
from image_cpp_cuda_tool.unet import UNet


print("Load DexiNed")
checkpoint_path = '/media/zhujun/0DFD06D20DFD06D2/MVS/FrameMVS/image_cpp_cuda_tool/checkpoints/BIPED/10/10_model.pth'

device = torch.device('cpu' if torch.cuda.device_count() == 0 else 'cuda')
model = DexiNed().to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

unet_model = UNet(n_channels=3, n_classes=1, bilinear=True).to(device)
unet_checkpoint_path = '/media/zhujun/0DFD06D20DFD06D2/MVS/FrameMVS/image_cpp_cuda_tool/checkpoints/10_model.pth'
unet_model.load_state_dict(torch.load(unet_checkpoint_path, map_location=device))
unet_model.eval()


def shuffle_mask(region_mask):
    unique_class=np.unique(region_mask)
    unique_class_new=np.arange(len(unique_class))
    old_class=unique_class.copy()
    np.random.shuffle(unique_class_new)
    region_mask_new=region_mask.copy()
    for old_class_i,old_class_v in enumerate(old_class):
        region_mask_new[region_mask==old_class_v]=unique_class_new[old_class_i]
    return region_mask_new

def plt_imgs(images_plot,show=True,save=False,cols=2):
    num_image=len(images_plot)
    rows=num_image//cols+1 if num_image%cols>0 else num_image//cols
    fig, ax = plt.subplots(rows, cols, figsize=(rows*6, cols*5), sharex=True, sharey=True)
    ax=ax.reshape(-1)
    for row in range(rows):
        for col in range(cols):
            image_idx=row*cols+col
            if image_idx>=num_image:
                ax[row*cols+col].set_axis_off()
            else:
                name,image,cmap=images_plot[image_idx]
                # import pdb;pdb.set_trace()
                ax[image_idx].imshow(image,cmap=cmap)
                ax[image_idx].set_title(name)
                ax[image_idx].set_axis_off()
    plt.tight_layout()
    if save:
        if not os.path.exists('tmp'):
            os.makedirs("tmp")
        plt.savefig('tmp/images_plot.png',dpi=500) #! figsize=(10,10) resolution=(500*10,500*10)=(5k,5k)
    if show:
        plt.show()

def get_edge_maps_tensor(raw_images_tensor):
    ishalf = raw_images_tensor.dtype is torch.float16
    with torch.no_grad():
        if raw_images_tensor.dim()==3:
            raw_images_tensor=raw_images_tensor[None]
        if raw_images_tensor.max()<1.5:
            raw_images_tensor = raw_images_tensor.float()*255.0
        if raw_images_tensor.shape[1]==3:
            raw_images_tensor = raw_images_tensor.permute(0,2,3,1)
        batch,h,w,c=raw_images_tensor.shape
        # raw_images_pro=raw_images_tensor.view(-1,3)-torch.as_tensor([103.939,116.779,123.68]).type_as(raw_images_tensor)
        raw_images_pro=raw_images_tensor-torch.as_tensor([103.939,116.779,123.68]).type_as(raw_images_tensor)[None,None,None,...]
        images_tensor=raw_images_pro.view(batch,h,w,c).permute(0,3,1,2)
        new_h = ((h // 16) + 1) * 16
        new_w = ((w // 16) + 1) * 16
        images_tensor_new=torch.nn.functional.interpolate(images_tensor,(new_h,new_w),mode='bilinear',align_corners=True)
        # preds = model(images_tensor_new)
        # edge_maps_tensor=torch.nn.functional.interpolate(torch.sigmoid(preds[6]),(h,w),mode='bilinear',align_corners=True)
        preds = unet_model(images_tensor_new)
        edge_maps_tensor=torch.nn.functional.interpolate(torch.sigmoid(preds[-1]),(h,w),mode='bilinear',align_corners=True)

        if ishalf:
            return edge_maps_tensor.half()
        else:

            return edge_maps_tensor

def edge_detector(raw_images_tensor):
    with torch.no_grad():
        if raw_images_tensor.dim()==3:
            raw_images_tensor=raw_images_tensor[None]
        if raw_images_tensor.max()<1.5:
            raw_images_tensor = raw_images_tensor.float()*255.0
        if raw_images_tensor.shape[1]==3:
            raw_images_tensor = raw_images_tensor.permute(0,2,3,1)
        batch,h,w,c=raw_images_tensor.shape
        raw_images_pro=raw_images_tensor.view(-1,3)-torch.as_tensor([103.939,116.779,123.68]).type_as(raw_images_tensor)
        images_tensor=raw_images_pro.view(batch,h,w,c).permute(0,3,1,2)
        new_h = ((h // 16) + 1) * 16
        new_w = ((w // 16) + 1) * 16
        images_tensor_new=torch.nn.functional.interpolate(images_tensor,(new_h,new_w),mode='bilinear',align_corners=True)
        # preds = model(images_tensor_new)
        # edge_maps_tensor=torch.nn.functional.interpolate(torch.sigmoid(preds[6]),(h,w),mode='bilinear',align_corners=True)
        preds = unet_model(images_tensor_new)
        edge_maps_tensor=torch.nn.functional.interpolate(torch.sigmoid(preds[-1]),(h,w),mode='bilinear',align_corners=True)
        return edge_maps_tensor

def get_region_seg_tensor(edge_maps_tensor,thred=0.1):
    edge_maps_tensor=edge_maps_tensor.squeeze()
    if edge_maps_tensor.dim()==2:
        edge_maps_tensor=edge_maps_tensor[None]
    batch_size,height,width=edge_maps_tensor.shape
    image_tensor=(edge_maps_tensor<=thred).float()
    mask=np.ones((batch_size,height,width),dtype=np.float32)-2
    # import pdb;pdb.set_trace()
    seg_masks_tensor=torch.from_numpy(mask).type_as(edge_maps_tensor)
    patchsize=3
    my_image_tool.image_seg(batch_size, height, width, patchsize, image_tensor, seg_masks_tensor)
    return seg_masks_tensor

def get_kpts_mask_tensor(edge_maps_tensor):
    edge_maps_tensor=edge_maps_tensor.squeeze()
    if edge_maps_tensor.dim()==2:
        edge_maps_tensor=edge_maps_tensor[None]
    batch_size,height,width=edge_maps_tensor.shape
    mask=np.ones((batch_size,height,width),dtype=np.float32)-2
    # import pdb;pdb.set_trace()
    kpts_mask_tensor=torch.from_numpy(mask).type_as(edge_maps_tensor)
    patchsize=3
    my_image_tool.kpts_selector(batch_size, height, width, patchsize, edge_maps_tensor, kpts_mask_tensor)
    return kpts_mask_tensor

def images_seg(raw_images,show=False,thred=0.1,edge=False,save=False):
    with torch.no_grad():
        if raw_images.dim()==3:
            raw_images=raw_images[None]
        if raw_images.max()<1.5:
            raw_images = raw_images.float()*255.0
        if raw_images.shape[1]==3:
            raw_images = raw_images.permute(0,2,3,1)
        batch,h,w,c=raw_images.shape
        raw_images_pro=raw_images.float().view(-1,3)-torch.as_tensor([103.939,116.779,123.68]).type_as(raw_images)
        images_tensor=raw_images_pro.view(batch,h,w,c).permute(0,3,1,2)
        new_h = ((h // 16) + 1) * 16
        new_w = ((w // 16) + 1) * 16
        images_tensor_new=torch.nn.functional.interpolate(images_tensor,(new_h,new_w),mode='bilinear',align_corners=True)
        start=time()
        # import pdb;pdb.set_trace()
        # preds = model(images_tensor_new.to(device))
        # print("Calculate time: "+str(time()-start))
        # edge_maps_tensor=torch.nn.functional.interpolate(torch.sigmoid(preds[6]),(h,w),mode='bilinear',align_corners=True)
        preds = unet_model(images_tensor_new)
        edge_maps_tensor=torch.nn.functional.interpolate(torch.sigmoid(preds[-1]),(h,w),mode='bilinear',align_corners=True)
        
        batch_size,ch,height,width=edge_maps_tensor.shape
        image_tensor=(edge_maps_tensor<=thred).squeeze(1).float()
        edge_maps_tensor=edge_maps_tensor.squeeze(1).float()
        mask=np.ones((batch_size,height,width),dtype=np.float32)-2
        # import pdb;pdb.set_trace()
        seg_masks_tensor=torch.from_numpy(mask).float().cuda()
        kpts_mask_tensor=torch.from_numpy(mask).float().cuda()
        patchsize=3
        my_image_tool.image_seg(batch_size, height, width, patchsize, image_tensor, seg_masks_tensor)
        my_image_tool.kpts_selector(batch_size, height, width, patchsize, edge_maps_tensor, kpts_mask_tensor)
        edge_maps=edge_maps_tensor.cpu().numpy()
        seg_masks=seg_masks_tensor.detach().cpu().numpy()
        kpts_mask=kpts_mask_tensor.detach().cpu().numpy()

        # import pdb;pdb.set_trace()
        # np.save("edge_maps.npy",edge_maps)
        # print('sdfasdvcasd')
        # kpts_mask=kpts_detect_cuda(edge_maps)
        #* (4, 1, 480, 640)
        # seg_masks=region_seg_cuda(edge_maps<=thred)
        if show:
            seg_masks_for_show=np.concatenate([shuffle_mask(img) for img in seg_masks],axis=1)
            raw_images_for_show=np.concatenate([img for img in raw_images.cpu().numpy()],axis=1)
            edge_maps_for_show=np.concatenate([img for img in edge_maps.squeeze()],axis=1)
            kpts_mask_for_show=np.concatenate([img for img in kpts_mask],axis=1)
            y,x=np.nonzero(kpts_mask_for_show>0)
            ref_kpts=np.hstack([x.reshape(-1,1),y.reshape(-1,1)])
            for junc in ref_kpts:
                cv2.circle(raw_images_for_show, tuple(junc.astype(int)), 9, (0,0,255), 2)
            # import pdb;pdb.set_trace()

            images_plot=[]
            images_plot.append(['orig_image',cv2.cvtColor(raw_images_for_show,cv2.COLOR_RGB2BGR),None])
            images_plot.append(['my_image_seg',seg_masks_for_show,'jet'])
            images_plot.append(['edge_maps',edge_maps_for_show,'jet'])
            images_plot.append(['kpts_mask',kpts_mask_for_show,'jet'])

            plt_imgs(images_plot,save=save,cols=2)
        # import pdb;pdb.set_trace()
        if edge:
            return seg_masks,kpts_mask,edge_maps.squeeze()
        return seg_masks,kpts_mask