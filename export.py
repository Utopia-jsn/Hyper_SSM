from ultralytics.models import YOLO 
# 加载训练好的模型
model = YOLO("runs/train/exp/weights/best.pt")
model.cuda() 
# 将模型转为onnx格式
success = model.export(format='onnx')