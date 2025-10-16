import torch
import torch.nn as nn
from torchsummary import summary
from ultralytics.nn.modules import SimpleStem, VSSBlock, VisionClueMerge, MANet
# print("After:", self.conv(x).size()) 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

t0 = SimpleStem.to(device)
summary(t0, (512, 8, 8))

