# python colmap_input.py --input_folder /home/zhujun/catkin_ws/src/r3live-master/r3live_output/data_for_mesh/dense \
# --output_folder /home/zhujun/MVS/PatchmatchNet/data/r3live/scan1
cd ACMMP_net
bash run.sh
cd ..

# ldd -r ACMMP_net/build/libacmmppy.so
# ldd -r  ACMMP_net/build/libsimpletools.so
# CHECKPOINT_FILE="./checkpoints/params_000007.ckpt"
# CHECKPOINT_FILE="./checkpoints/log/model_000007.ckpt"

# # test on DTU's evaluation set
# DTU_TESTING="data/r3live/"
# DTU_TESTING="/home/zhujun/ubuntu_data/MVS/PatchmatchNet/data_for_mesh"
# DTU_TESTING="/home/zhujun/catkin_ws/src/r3live-master/r3live_output/data_for_mesh_front"
# python local_mvs.py --input_folder=$DTU_TESTING --output_folder=$DTU_TESTING/PatchmatchNet_depthguide \
# --checkpoint_path $CHECKPOINT_FILE --num_views 1 --image_max_dim 1600 --geo_mask_thres 3 --photo_thres 0.8 "$@"

# watch -n 1 nvidia-smi
#* 编译cpp-cuda
# cd image_cpp_cuda_tool/MyCppCudaTool && bash run.sh && cd -

# python local_mvs.py
python pc_img_fusion.py
# sudo apt-get install ros-melodic-desktop-full ros-melodic-desktop  ros-melodic-perception  ros-melodic-simulators ros-melodic-urdf-sim-tutorial


# watch -n 1 nvidia-smi
#* 编译cpp-cuda
# cd image_cpp_cuda_tool/MyCppCudaTool && bash run.sh && cd -

# 推送现有文件夹
# cd existing_folder
# git init
# git remote add origin git@gitcode.net:qq_41093957/framemvs.git
# git add .
# git commit -m "Initial commit"
# git push -u origin master
# git push origin master
