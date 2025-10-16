import torch
import segmentation_models_pytorch as smp
# from ghostunet1 import ghostunet
from torchstat import stat
# from model1 import swUnet
from torchsummary import summary
# from vision_transformer import SwinUnet
# from TransUnet import get_transNet
# from nets.frcnn import FasterRCNN
from ultralytics import YOLO  

# 需要使用device来指定网络在GPU还是CPU运行
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型总大小为：{:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)

def main():
    # model = FasterRCNN(num_classes=16, backbone='vgg')
    model = YOLO(model='ultralytics/cfg/models/mamba-yolo/hyper-mamba24-B.yaml')
    getModelSize(model)
    model.cuda()
if __name__ == '__main__':
    main()
