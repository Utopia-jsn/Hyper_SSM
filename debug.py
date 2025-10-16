import torch
import torch.nn as nn

# 创建一个示例特征图，尺寸为 [batch_size, channels, height, width]
feature_map = torch.randn(1, 512, 8, 8)

# 创建上采样层
upsample_layer = nn.Upsample(scale_factor=2, mode='nearest')

# 应用上采样
upsampled_feature_map = upsample_layer(feature_map)

# 输出上采样后的特征图尺寸
print(upsampled_feature_map.size())