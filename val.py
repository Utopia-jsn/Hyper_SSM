# from ultralytics.models import RTDETR
# import os

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
 
# if __name__ == '__main__':
#     model = RTDETR(model='./runs/train/exp/weights/best.pt')
#     model.val(data='wsodd_usv.yaml', split='val', batch=16, device='0', project='runs/val', name='exp',
#               half=False,)


from ultralytics import YOLO

if __name__=="__main__":
    
    pth_path=r"./runs/train/exp13/weights/best.pt"
    
    # 独立验证
    model = YOLO(pth_path)
      # 如果不设置数据，它将使用model.pt中的数据集相关yaml文件。
    metrics = model.val(data='WTDataset.yaml',project='runs/val', batch=1, iou= 0.5, half=True, max_det= 100, dnn= True)
    
    # metrics.box.map    # map50-95
    # metrics.box.map50  # map50
    # metrics.box.map75  # map75
    # metrics.box.maps   # a list contains map50-95 of each category
