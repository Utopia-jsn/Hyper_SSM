from ultralytics import YOLO  
import argparse
import os

# train.py
import sys
sys.path.append("/root/project/Mamba-YOLO-main")

if __name__ == '__main__':
    model = YOLO(model='ultralytics/cfg/models/mamba-yolo/hyper-ssm-B.yaml')
    # model.load('runs/train/exp7/weights/best.pt')  # Uncomment and specify your pre-trained weights path
    model.train(data='WTDataset.yaml', 
                epochs=100, 
                batch=8, 
                seed=42,
                lr0=0.01,
                lrf=0.01,
                device='0',
                optimizer='SGD',
                imgsz=640, 
                workers=128, 
                amp=True,
                project='runs/train', 
                name='exp')
    os.system("/usr/bin/shutdown")

# from ultralytics import YOLO
# import argparse
# import os

# ROOT = '/root/project'
# print(ROOT)

# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data', type=str, default=os.path.normpath(os.path.join(ROOT, 'Mamba-YOLO-main/wsodd_usv.yaml')), help='dataset.yaml path')
#     parser.add_argument('--config', type=str, default=os.path.normpath(os.path.join(ROOT,'Mamba-YOLO-main/ultralytics/cfg/models/mamba-yolo/hyper-mamba4-B.yaml')), help='model path(s)')
#     parser.add_argument('--batch_size', type=int, default=8, help='batch size')
#     parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
#     parser.add_argument('--task', default='train', help='train, val, test, speed or study')
#     parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--workers', type=int, default=128, help='max dataloader workers (per RANK in DDP mode)')
#     parser.add_argument('--epochs', type=int, default=100)
#     parser.add_argument('--optimizer', default='SGD', help='SGD, Adam, AdamW')
#     parser.add_argument('--amp', action='store_true', help='open amp')
#     parser.add_argument('--project', default=ROOT + '/Mamba-YOLO-main/runs/train', help='save to project/name')
#     parser.add_argument('--name', default='exp', help='save to project/name')
#     parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
#     parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
#     opt = parser.parse_args()
#     return opt


# if __name__ == '__main__':
#     opt = parse_opt()
#     task = opt.task
#     args = {
#         "data": opt.data,
#         "epochs": opt.epochs,
#         "workers": opt.workers,
#         "batch_size": opt.batch_size,  # 修改为 batch_size
#         "optimizer": opt.optimizer,
#         "device": opt.device,
#         "amp": opt.amp,
#         "project": opt.project,
#         "name": opt.name,
#         # 添加其他需要的参数...
#     }
#     model_conf = opt.config

#     task_type = {
#         "train": YOLO(model_conf).train(**args),
#         "val": YOLO(model_conf).val(**args),
#         "test": YOLO(model_conf).test(**args),
#     }
#     task_type.get(task)

