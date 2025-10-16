

from ultralytics import YOLO
 
if __name__=="__main__":
    
    pth_path=r"./runs/train/mambayolo/weights/best.pt"
 
    test_path=r"/root/autodl-tmp/project/WSODD USV_dataset/images/test"
    # Load a model
    #model = YOLO('yolov8n.pt')  # load an official model
    model = YOLO(pth_path)  # load a custom model
    
    print(model.info())
    # Predict with the model
    results = model(test_path, save=True, conf=0.5, batch=1)   # predict on an image


