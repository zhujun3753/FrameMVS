#* 运行python setup.py develop得到编译后的so文件保存在该目录下（如果用python setup.py install则保存在系统路径中）
python setup.py develop
# python python_code/my_tool_example.py

# nm -gDC MyCppCudaTool/my_image_cpp_cuda_tool/my_image_tool.cpython-39-x86_64-linux-gnu.so | grep -i image_seg_cpp