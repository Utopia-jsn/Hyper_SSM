# from ultralytics import RTDETR

# if __name__ == "__main__":
    
#     pth_path = r"./runs/train/exp/weights/best.pt"
#     test_path = r"./WSODD USV_dataset/images/test"
    
#     # Load a model
#     model = RTDETR(pth_path)  # load a custom model
    
#     # Predict with the model
#     # Set the batch size to a specific value, e.g., 16
#     results = model(test_path, save=True, conf=0.5, batch=1)  # predict on an image with a batch size of 16


from ultralytics import YOLO

if __name__=="__main__":
    
    pth_path = r"./runs/train/exp2/weights/best.pt"
    test_path = r"/root/autodl-tmp/WTDataset_txt_min_noise/images/test"
    # Load a model
    #model = YOLO('yolov8n.pt')  # load an official model
    model = YOLO(pth_path)  # load a custom model
 
    # Predict with the model
    results = model.predict(test_path, imgsz=640,save=True, conf=0.5, device=0, batch=1)   # predict on an image



    


