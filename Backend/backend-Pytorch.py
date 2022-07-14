import pyrebase
import time
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from config import *

def load_model():
    my_model = torch.load(model_path_pytorch , map_location=DEVICE)
    my_model.eval()
    return my_model

def preprocess():
    img = Image.open(img_path)
    test_transforms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
    image_tensor = test_transforms(img)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE) 
    return image_tensor

def predict(model , test_data):
    output_tensor = model(test_data)
    output_tensor = output_tensor.max(1).indices
    index = output_tensor.item()
    result = label_dict[index]
    return result

def start_prediction():
    test_data = preprocess()
    result = predict(model , test_data)
    return result

def stream_handler(message):
    cloud_path = "test.jpg"
    if message["data"] == 1:
        print(f"\nProcessing...")
        begin = time.time()
        storage.child(cloud_path).download( Path(dir_path).joinpath("data").joinpath("test.jpg").as_posix() )
        result = start_prediction() # 進行影像辨識處理區段，把顯示結果填到result
        if result != "": # 有辨識結果為2
            print(f"Result : {result}")
            data = {"pictureStatus":2}
            data["result"] = result
            db.update(data)
        else: # 無法辨識結果為3
            print(f"Result : Unknown")
            data = {"result":"", "pictureStatus":3}
            db.update(data)
        end = time.time()
        print(f"Time Elapsed : {round(end-begin,2)}s")

if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model()
    firebase = pyrebase.initialize_app(config)
    db = firebase.database()
    storage = firebase.storage()
    my_stream = db.child("pictureStatus").stream(stream_handler)
    print("Backend Running...")