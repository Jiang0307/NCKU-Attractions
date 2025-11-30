import pyrebase
import time
import numpy as np
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
from tensorflow.keras.models import load_model
from pathlib import Path
from PIL import Image
from config import *

def download_model_from_storage(storage, model_storage_path, local_path):
    """從 Firebase Storage 下載模型檔案"""
    try:
        print(f"正在從 Firebase Storage 下載模型: {model_storage_path}")
        storage.child(model_storage_path).download(filename=local_path)
        print(f"模型下載完成: {local_path}")
        return True
    except Exception as e:
        print(f"下載模型時發生錯誤: {e}")
        return False

def load_model_from_file(model_path, storage=None):
    """載入模型，如果本地不存在則從 Firebase Storage 下載"""
    # 檢查本地模型檔案是否存在
    if not Path(model_path).exists():
        if storage is None:
            raise FileNotFoundError(f"模型檔案不存在: {model_path}，且未提供 storage 物件")
        
        # 確保 model 目錄存在
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 從 Firebase Storage 下載模型
        if not download_model_from_storage(storage, model_storage_path_keras, model_path):
            raise FileNotFoundError(f"無法從 Firebase Storage 下載模型: {model_storage_path_keras}")
    
    # 直接載入模型（模型檔案應該已經在映像中或已下載）
    return load_model(model_path)

def preprocess():
    PIL_img = Image.open(img_path)
    resized_img = PIL_img.resize((224, 224))
    img = np.asarray(resized_img)
    test_data = np.expand_dims(img , axis=0)
    return test_data

def predict(test_data):
    result_array = model.predict(test_data)
    result = label_dict[np.argmax(result_array)]
    return result

def start_prediction():
    test_data = preprocess()
    result = predict(test_data)
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
    # 載入模型（如果映像中包含模型檔案，直接載入；否則從 Firebase Storage 下載）
    print("正在載入模型...")
    
    # 先檢查本地是否有模型檔案
    if Path(model_path_keras).exists():
        print("找到本地模型檔案，直接載入")
        model = load_model(model_path_keras)
    else:
        # 如果沒有，從 Firebase Storage 下載
        print("本地模型不存在，從 Firebase Storage 下載...")
        firebase = pyrebase.initialize_app(config)
        storage = firebase.storage()
        model = load_model_from_file(model_path_keras, storage=storage)
    
    print("模型載入完成")
    
    # 初始化 Firebase
    firebase = pyrebase.initialize_app(config)
    db = firebase.database()
    storage = firebase.storage()
    
    # 開始監聽 Firebase Realtime Database
    my_stream = db.child("pictureStatus").stream(stream_handler)
    print("Backend Running...")