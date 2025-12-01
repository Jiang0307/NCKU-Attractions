import pyrebase
import time
import numpy as np
import requests
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
from tensorflow.keras.models import load_model
from pathlib import Path
from PIL import Image
from config import *

def download_model_from_url(url, local_path):
    """從 URL 下載模型檔案（使用 requests）"""
    try:
        print(f"正在從 URL 下載模型...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # 確保目錄存在
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 下載檔案
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        print(f"模型下載完成: {local_path}")
        return True
    except Exception as e:
        print(f"下載模型時發生錯誤: {e}")
        return False

def download_model_from_storage(storage, model_storage_path, local_path):
    """從 Firebase Storage 下載模型檔案（使用 pyrebase）"""
    try:
        print(f"正在從 Firebase Storage 下載模型: {model_storage_path}")
        # path 是本地保存的目錄，filename 是檔名
        local_dir = Path(local_path).parent.as_posix()
        filename = Path(local_path).name
        storage.child(model_storage_path).download(path=local_dir, filename=filename)
        print(f"模型下載完成: {local_path}")
        return True
    except Exception as e:
        print(f"下載模型時發生錯誤: {e}")
        return False

def load_model_from_file(model_path, storage=None, download_url=None):
    """載入模型，如果本地不存在則下載"""
    # 檢查本地模型檔案是否存在
    if not Path(model_path).exists():
        # 優先使用 URL 下載（更可靠）
        if download_url:
            print("使用 URL 下載模型...")
            if not download_model_from_url(download_url, model_path):
                raise FileNotFoundError(f"無法從 URL 下載模型: {download_url}")
        elif storage:
            # 備用方案：使用 pyrebase 下載
            print("使用 Firebase Storage 下載模型...")
            if not download_model_from_storage(storage, model_storage_path_keras, model_path):
                raise FileNotFoundError(f"無法從 Firebase Storage 下載模型: {model_storage_path_keras}")
        else:
            raise FileNotFoundError(f"模型檔案不存在: {model_path}，且未提供下載方式")
    
    # 載入模型（使用與測試腳本相同的方式）
    try:
        print(f"正在載入模型: {model_path}")
        # 嘗試標準載入
        model = load_model(model_path)
        print("模型載入成功")
        return model
    except Exception as e:
        print(f"標準載入失敗: {e}")
        # 嘗試使用 compile=False 載入（可能解決某些版本兼容問題）
        try:
            print("嘗試使用 compile=False 載入模型...")
            model = load_model(model_path, compile=False)
            print("模型載入成功（使用 compile=False）")
            return model
        except Exception as e2:
            print(f"使用 compile=False 也失敗: {e2}")
            print("這可能是因為：")
            print("1. 模型檔案損壞或不完整")
            print("2. TensorFlow/Keras 版本不兼容")
            print("3. 模型架構問題")
            print("4. Firebase Storage 中的模型與本地模型不同")
            # 如果載入失敗，刪除檔案以便重新下載
            if Path(model_path).exists():
                try:
                    Path(model_path).unlink()
                    print(f"已刪除損壞的檔案，下次將重新下載")
                except:
                    pass
            raise e2

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
        local_dir = Path(dir_path).joinpath("data").as_posix()
        storage.child(cloud_path).download(path=local_dir, filename="test.jpg")
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
    # 載入模型
    print("正在載入模型...")
    
    # 先檢查本地是否有模型檔案
    if Path(model_path_keras).exists():
        print("找到本地模型檔案，直接載入")
        try:
            model = load_model(model_path_keras)
        except Exception as e:
            print(f"載入本地模型失敗: {e}，將重新下載")
            # 刪除損壞的檔案
            Path(model_path_keras).unlink()
            # 從 URL 或 Firebase Storage 下載
            firebase = pyrebase.initialize_app(config)
            storage = firebase.storage()
            model = load_model_from_file(
                model_path_keras, 
                storage=storage, 
                download_url=model_download_url_keras if model_download_url_keras else None
            )
    else:
        # 如果沒有，從 URL 或 Firebase Storage 下載
        print("本地模型不存在，開始下載...")
        firebase = pyrebase.initialize_app(config)
        storage = firebase.storage()
        model = load_model_from_file(
            model_path_keras, 
            storage=storage, 
            download_url=model_download_url_keras if model_download_url_keras else None
        )
    
    print("模型載入完成")
    
    # 初始化 Firebase
    firebase = pyrebase.initialize_app(config)
    db = firebase.database()
    storage = firebase.storage()
    
    # 開始監聽 Firebase Realtime Database
    my_stream = db.child("pictureStatus").stream(stream_handler)
    print("Backend Running...")