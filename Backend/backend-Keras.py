import pyrebase
import time
import numpy as np
import os
import h5py
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
from tensorflow.keras.models import load_model
from pathlib import Path
from PIL import Image
from config import *

def verify_h5_file(file_path):
    """驗證 H5 檔案是否有效"""
    try:
        with h5py.File(file_path, 'r') as f:
            # 嘗試打開檔案，如果能打開就表示檔案有效
            return True
    except Exception as e:
        print(f"H5 檔案驗證失敗: {e}")
        return False

def download_model_from_storage(storage, model_storage_path, local_path, max_retries=3):
    """從 Firebase Storage 下載模型檔案（帶重試機制）"""
    for attempt in range(max_retries):
        try:
            print(f"正在從 Firebase Storage 下載模型: {model_storage_path} (嘗試 {attempt + 1}/{max_retries})")
            
            # 確保目錄存在
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            
            # 下載檔案
            storage.child(model_storage_path).download(local_path)
            
            # 驗證檔案是否存在且有內容
            if not Path(local_path).exists():
                raise FileNotFoundError(f"下載後檔案不存在: {local_path}")
            
            file_size = os.path.getsize(local_path)
            print(f"檔案下載完成: {local_path}, 大小: {file_size} bytes")
            
            if file_size == 0:
                raise ValueError("下載的檔案大小為 0")
            
            # 驗證 H5 檔案格式
            if not verify_h5_file(local_path):
                raise ValueError("下載的檔案不是有效的 H5 格式")
            
            print(f"模型下載並驗證成功: {local_path}")
            return True
            
        except Exception as e:
            print(f"下載嘗試 {attempt + 1} 失敗: {e}")
            # 如果檔案存在但損壞，刪除它以便重新下載
            if Path(local_path).exists():
                try:
                    os.remove(local_path)
                    print(f"已刪除損壞的檔案: {local_path}")
                except:
                    pass
            
            if attempt == max_retries - 1:
                print(f"所有下載嘗試都失敗了")
                return False
            
            time.sleep(2)  # 等待後重試
    
    return False

def load_model_from_file(model_path, storage=None):
    """載入模型，如果本地不存在則從 Firebase Storage 下載"""
    # 檢查本地模型檔案是否存在且有效
    if Path(model_path).exists():
        print(f"找到本地模型檔案: {model_path}")
        # 驗證檔案是否有效
        if not verify_h5_file(model_path):
            print(f"本地模型檔案損壞，將重新下載")
            os.remove(model_path)
        else:
            print(f"本地模型檔案有效，直接載入")
            try:
                return load_model(model_path)
            except Exception as e:
                print(f"載入本地模型失敗: {e}，將重新下載")
                if Path(model_path).exists():
                    os.remove(model_path)
    
    # 如果本地沒有有效檔案，從 Firebase Storage 下載
    if storage is None:
        raise FileNotFoundError(f"模型檔案不存在: {model_path}，且未提供 storage 物件")
    
    print(f"本地模型不存在或損壞，從 Firebase Storage 下載...")
    
    # 確保 model 目錄存在
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 從 Firebase Storage 下載模型
    if not download_model_from_storage(storage, model_storage_path_keras, model_path):
        raise FileNotFoundError(f"無法從 Firebase Storage 下載模型: {model_storage_path_keras}")
    
    # 下載成功後載入模型
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
    # 先初始化 Firebase 以獲取 storage 物件
    firebase = pyrebase.initialize_app(config)
    db = firebase.database()
    storage = firebase.storage()
    
    # 載入模型（如果需要會從 Firebase Storage 下載）
    print("正在載入模型...")
    model = load_model_from_file(model_path_keras, storage=storage)
    print("模型載入完成")
    
    # 開始監聽 Firebase Realtime Database
    my_stream = db.child("pictureStatus").stream(stream_handler)
    print("Backend Running...")