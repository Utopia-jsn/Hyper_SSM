import torch.nn as nn
import torch
from thop import profile
# from nets.classifier import Resnet50RoIHead, VGG16RoIHead
# from nets.resnet50 import resnet50
# from nets.rpn import RegionProposalNetwork
# from nets.vgg16 import decom_vgg16
from torchprofile import profile_macs
# from nets.frcnn import FasterRCNN
from ultralytics import YOLO  

    
model = YOLO(model='ultralytics/cfg/models/mamba-yolo/hyper-mamba24-B.yaml')
# 创建正确范围的输入张量
input_tensor = torch.rand(1, 3, 640, 640)  # 0-1范围

# 计算FLOPs
with torch.no_grad():
    macs, params = profile(model, inputs=(input_tensor,), verbose=False)
    
# 计算 FLOPs 和参数量
macs = profile_macs(model, input_tensor)

# 转换为 GFLOPs
gflops = macs * 2 / 1e9  # 乘以 2 是因为每个 MAC 包含一次乘法和一次加法
print(f"Total GFLOPs: {gflops:.2f}")

# results = estimate_gflops(model)
# if results:
#     print(f"MACs: {results['MACs']} G")
#     print(f"Params: {results['Params']} M")
#     print(f"GFLOPs: {results['GFLOPs']}")