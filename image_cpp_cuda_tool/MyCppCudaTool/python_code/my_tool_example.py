
import torch
from my_cpp_cuda_tool import my_tool
import numpy as np
import matplotlib.pyplot as plt
from time import time


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
        plt.savefig('seg_comp.png',dpi=500) #! figsize=(10,10) resolution=(500*10,500*10)=(5k,5k)
    if show:
        plt.show()

def shuffle_mask(region_mask):
    unique_class=np.unique(region_mask)
    unique_class_new=np.arange(len(unique_class))
    old_class=unique_class.copy()
    np.random.shuffle(unique_class_new)
    region_mask_new=region_mask.copy()
    for old_class_i,old_class_v in enumerate(old_class):
        region_mask_new[region_mask==old_class_v]=unique_class_new[old_class_i]
    return region_mask_new


if __name__ == '__main__':
    edge_maps = np.load("python_code/edge_maps.npy")
    image=edge_maps<=0.1
    if torch.cuda.device_count() == 0:
        print("No cuda !! Exit")
        exit()
    batch_size,ch,height,width=image.shape
    if ch==3:
        print("Only [N,1,H,W] is available now !")
        raise NotImplementedError
    patchsize=3
    image_tensor=torch.from_numpy(image.squeeze(1)).float().cuda()
    edge_maps_tensor=torch.from_numpy(edge_maps.squeeze(1)).float().cuda()
    mask=np.ones((batch_size,height,width),dtype=np.float32)-2
    # import pdb;pdb.set_trace()
    seg_masks_tensor=torch.from_numpy(mask).float().cuda()
    kpts_mask_tensor=torch.from_numpy(mask).float().cuda()
    start=time()
    my_tool.image_seg(batch_size, height, width, patchsize, image_tensor, seg_masks_tensor)
    my_tool.kpts_selector(batch_size, height, width, patchsize, edge_maps_tensor, kpts_mask_tensor)
    seg_masks=seg_masks_tensor.detach().cpu().numpy()
    kpts_mask=kpts_mask_tensor.detach().cpu().numpy()
    print("Calculate time: "+str(time()-start))
    show=True
    if show:
        seg_masks_for_show=np.concatenate([shuffle_mask(img) for img in seg_masks],axis=1)
        edge_maps_for_show=np.concatenate([img for img in edge_maps.squeeze()],axis=1)
        kpts_mask_for_show=np.concatenate([img for img in kpts_mask],axis=1)
        images_plot=[]
        images_plot.append(['my_image_seg',seg_masks_for_show,'jet'])
        images_plot.append(['edge_maps',edge_maps_for_show,'jet'])
        images_plot.append(['kpts_mask',kpts_mask_for_show,'jet'])
        # import pdb;pdb.set_trace()

        plt_imgs(images_plot,cols=2)
