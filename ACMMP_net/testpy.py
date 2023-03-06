from matplotlib import image
import torch
import cv2
print("Load lib")
torch.ops.load_library("build/libacmmppy.so")
# acmmp = torch.ops.acmmp
# # print(torch.ops.acmmp.maintest)
# image_filename = '/home/zhujun/catkin_ws/src/r3live-master/r3live_output/data_for_mesh_front/acmmp/images/00000000.jpg'
# image_np = cv2.imread(image_filename,1)[:,:,[2,1,0]] #* 1 color 0 gray -1 unchange
# image_np = cv2.imread(image_filename,0)[...,None] #* 1 color 0 gray -1 unchange
# image_torch = torch.from_numpy(image_np).float().cuda()[None].repeat(5,1,1,1).permute(0,3,1,2)
# print(image_torch.shape) #* torch.Size([5, 3, 1002, 1253])
# # import pdb;pdb.set_trace()
# # print(image_torch[:,:,:4,:4])
# acmmp.tensorToMat(image_torch)
torch.classes.load_library("build/libacmmppy.so")
# Params = torch.classes.acmmp.Params()
# ACMMP = torch.classes.acmmp.ACMMP()
# Params.test()
import pdb;pdb.set_trace()
# acmmp.maintest()
