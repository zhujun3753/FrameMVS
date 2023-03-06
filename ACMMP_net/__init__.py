import torch
print("Load lib")
torch.ops.load_library("/media/zhujun/0DFD06D20DFD06D2/MVS/FrameMVS/ACMMP_net/build/libacmmppy.so")
torch.ops.load_library("/media/zhujun/0DFD06D20DFD06D2/MVS/FrameMVS/ACMMP_net/build/libsimpletools.so")
simpletools = torch.ops.simple_tools
acmmp = torch.ops.acmmp
torch.classes.load_library("/media/zhujun/0DFD06D20DFD06D2/MVS/FrameMVS/ACMMP_net/build/libacmmppy.so")
Params = torch.classes.acmmp.Params()
ACMMP = torch.classes.acmmp.ACMMP()
# import pdb;pdb.set_trace()
Global_map = torch.classes.acmmp.Global_map()

print("Load lib end")

# Params.test()