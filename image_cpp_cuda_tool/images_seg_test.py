import os
import cv2
from torch.utils.data import DataLoader
from model import DexiNed
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch
import numpy as np
from images_seg import images_seg

def image_normalization(img, img_min=0, img_max=255, epsilon=1e-12):
    img = np.float32(img)
    img = (img - np.min(img)) * (img_max - img_min) / ((np.max(img) - np.min(img)) + epsilon) + img_min
    return img

class TestDataset(Dataset):
    def __init__(self,data_root,mean_bgr=[103.939,116.779,123.68],):
        self.data_root = data_root
        self.mean_bgr = mean_bgr
        self.data_index =[os.path.join(self.data_root, f) for f in  sorted(os.listdir(self.data_root)) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        image_path = self.data_index[idx]
        img_name = os.path.basename(image_path)
        file_name = os.path.splitext(img_name)[0] + ".png"
        image = cv2.imread(os.path.join('', image_path), cv2.IMREAD_COLOR)
        img=image
        # img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
        if img.shape[0] % 16 != 0 or img.shape[1] % 16 != 0:
            img_width = ((img.shape[1] // 16) + 1) * 16
            img_height = ((img.shape[0] // 16) + 1) * 16
            img = cv2.resize(img, (img_width, img_height))
        im_shape = [img.shape[0], img.shape[1]]
        img = np.array(img, dtype=np.float32)
        img -= self.mean_bgr
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()
        return dict(images=img,raw_images=image, file_names=file_name, image_shape=im_shape,img_path=image_path)

def main():
    img_path='img/test/data/BIPEDv2/BIPED/edges/imgs/test/rgbr'
    img_path='/home/zhujun/MVS/data/scannet_3dvnet/scans_test/scene0708_00/orig_img'
    img_path='/home/zhujun/MVS/data/strayscaner/scans_test/8abc938498/color'
    img_path='/home/zhujun/MVS/frame_mvs/images_seg/orig_img'
    img_path = '/home/zhujun/ubuntu_data/MVS/PatchmatchNet/data_for_mesh/images'
    img_path = '/home/zhujun/catkin_ws/src/r3live-master/r3live_output/data_for_mesh_end/images'

    # img_path='img/test/data'
    dataset_val = TestDataset(img_path)
    dataloader_val = DataLoader(dataset_val,batch_size=2,shuffle=False,num_workers=16)

    for batch_id, sample_batched in enumerate(dataloader_val):
        print(batch_id)
        raw_images=sample_batched['raw_images'] #* torch.Size([4, 480, 640, 3])
        seg_masks=images_seg(raw_images,show=True)

if __name__ == '__main__':
    main()

